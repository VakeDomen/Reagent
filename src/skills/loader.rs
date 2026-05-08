use std::{
    collections::{HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
};

use super::{Skill, SkillLoadError, SkillResource, SkillResourceKind};

pub fn load_skill(path: impl AsRef<Path>) -> Result<Skill, SkillLoadError> {
    let skill_md = resolve_skill_file(path.as_ref())?;
    let skill_md = skill_md.canonicalize()?;
    let root = skill_md
        .parent()
        .ok_or_else(|| {
            SkillLoadError::InvalidPath(format!(
                "Skill file has no parent directory: {}",
                skill_md.display()
            ))
        })?
        .to_path_buf();

    let raw_content = fs::read_to_string(&skill_md)?;
    let (frontmatter, instructions) = split_frontmatter(&skill_md, &raw_content)?;
    let parsed = parse_frontmatter(&skill_md, frontmatter)?;
    validate_skill_name(&skill_md, &root, &parsed.name)?;
    let instructions = instructions.to_string();

    Ok(Skill {
        name: parsed.name,
        description: parsed.description,
        resources: discover_resources(&root, &skill_md)?,
        root,
        skill_md,
        raw_content,
        instructions,
        license: parsed.license,
        compatibility: parsed.compatibility,
        metadata: parsed.metadata,
        allowed_tools: parsed.allowed_tools,
    })
}

pub fn load_skill_collection(path: impl AsRef<Path>) -> Result<Vec<Skill>, SkillLoadError> {
    let root = path.as_ref().canonicalize()?;
    if !root.is_dir() {
        return Err(SkillLoadError::InvalidPath(format!(
            "Skill collection must be a directory: {}",
            root.display()
        )));
    }

    let mut child_dirs = Vec::new();
    for entry in fs::read_dir(&root)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() || is_hidden_path(&path) {
            continue;
        }
        if find_skill_file_in_dir(&path).is_some() {
            child_dirs.push(path);
        }
    }

    child_dirs.sort();

    let mut skills = Vec::new();
    for child in child_dirs {
        skills.push(load_skill(child)?);
    }

    if skills.is_empty() {
        return Err(SkillLoadError::NoSkillsInCollection(root));
    }

    Ok(skills)
}

pub fn load_skill_sources(
    skill_paths: &[PathBuf],
    collection_paths: &[PathBuf],
) -> Result<Vec<Skill>, SkillLoadError> {
    let mut roots = HashSet::new();
    let mut names = HashSet::new();
    let mut skills = Vec::new();

    for path in skill_paths {
        push_unique_skill(load_skill(path)?, &mut roots, &mut names, &mut skills)?;
    }

    for path in collection_paths {
        for skill in load_skill_collection(path)? {
            push_unique_skill(skill, &mut roots, &mut names, &mut skills)?;
        }
    }

    Ok(skills)
}

fn push_unique_skill(
    skill: Skill,
    roots: &mut HashSet<PathBuf>,
    names: &mut HashSet<String>,
    skills: &mut Vec<Skill>,
) -> Result<(), SkillLoadError> {
    if !names.insert(skill.name.clone()) {
        return Err(SkillLoadError::DuplicateSkillName(skill.name));
    }
    if roots.insert(skill.root.clone()) {
        skills.push(skill);
    }
    Ok(())
}

fn resolve_skill_file(path: &Path) -> Result<PathBuf, SkillLoadError> {
    if path.is_file() {
        let file_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default();
        if file_name.eq_ignore_ascii_case("skill.md") {
            return Ok(path.to_path_buf());
        }
        return Err(SkillLoadError::InvalidPath(format!(
            "Skill file path must point to SKILL.md or skill.md: {}",
            path.display()
        )));
    }

    if path.is_dir() {
        return find_skill_file_in_dir(path)
            .ok_or_else(|| SkillLoadError::MissingSkillFile(path.to_path_buf()));
    }

    Err(SkillLoadError::InvalidPath(format!(
        "Skill path does not exist: {}",
        path.display()
    )))
}

fn find_skill_file_in_dir(path: &Path) -> Option<PathBuf> {
    let upper = path.join("SKILL.md");
    if upper.is_file() {
        return Some(upper);
    }

    let lower = path.join("skill.md");
    if lower.is_file() {
        return Some(lower);
    }

    None
}

fn split_frontmatter<'a>(
    path: &Path,
    content: &'a str,
) -> Result<(&'a str, &'a str), SkillLoadError> {
    let mut lines = content.lines();
    if lines.next() != Some("---") {
        return Err(SkillLoadError::MissingFrontmatter(path.to_path_buf()));
    }

    let frontmatter_start = content
        .find('\n')
        .ok_or_else(|| SkillLoadError::MissingFrontmatter(path.to_path_buf()))?
        + 1;

    let mut offset = frontmatter_start;
    for line in content[frontmatter_start..].lines() {
        let line_len = line.len();
        if line == "---" {
            let body_start = offset + line_len;
            let body_start = if content[body_start..].starts_with("\r\n") {
                body_start + 2
            } else if content[body_start..].starts_with('\n') {
                body_start + 1
            } else {
                body_start
            };
            return Ok((&content[frontmatter_start..offset], &content[body_start..]));
        }
        offset += line_len;
        if content[offset..].starts_with("\r\n") {
            offset += 2;
        } else if content[offset..].starts_with('\n') {
            offset += 1;
        }
    }

    Err(SkillLoadError::MissingFrontmatter(path.to_path_buf()))
}

#[derive(Default)]
struct ParsedFrontmatter {
    name: String,
    description: String,
    license: Option<String>,
    compatibility: Option<String>,
    metadata: HashMap<String, String>,
    allowed_tools: Option<String>,
}

fn parse_frontmatter(path: &Path, frontmatter: &str) -> Result<ParsedFrontmatter, SkillLoadError> {
    let mut parsed = ParsedFrontmatter::default();
    let mut in_metadata = false;

    for line in frontmatter.lines() {
        if line.trim().is_empty() || line.trim_start().starts_with('#') {
            continue;
        }

        let is_indented = line.starts_with(' ') || line.starts_with('\t');
        if in_metadata && is_indented {
            if let Some((key, value)) = split_key_value(line.trim()) {
                parsed
                    .metadata
                    .insert(key.to_string(), normalize_yaml_scalar(value));
            }
            continue;
        }

        in_metadata = false;
        let Some((key, value)) = split_key_value(line) else {
            continue;
        };

        match key {
            "name" => parsed.name = normalize_yaml_scalar(value),
            "description" => parsed.description = normalize_yaml_scalar(value),
            "license" => parsed.license = Some(normalize_yaml_scalar(value)),
            "compatibility" => parsed.compatibility = Some(normalize_yaml_scalar(value)),
            "allowed-tools" => parsed.allowed_tools = Some(normalize_yaml_scalar(value)),
            "metadata" => in_metadata = true,
            _ => {}
        }
    }

    if parsed.name.is_empty() {
        return Err(SkillLoadError::MissingField {
            path: path.to_path_buf(),
            field: "name",
        });
    }
    if parsed.description.is_empty() {
        return Err(SkillLoadError::MissingField {
            path: path.to_path_buf(),
            field: "description",
        });
    }
    if parsed.description.chars().count() > 1024 {
        return Err(SkillLoadError::InvalidField {
            path: path.to_path_buf(),
            field: "description",
            message: "must be 1024 characters or fewer".into(),
        });
    }
    if let Some(compatibility) = &parsed.compatibility {
        if compatibility.chars().count() > 500 {
            return Err(SkillLoadError::InvalidField {
                path: path.to_path_buf(),
                field: "compatibility",
                message: "must be 500 characters or fewer".into(),
            });
        }
    }

    Ok(parsed)
}

fn split_key_value(line: &str) -> Option<(&str, &str)> {
    let (key, value) = line.split_once(':')?;
    Some((key.trim(), value.trim()))
}

fn normalize_yaml_scalar(value: &str) -> String {
    let value = value.trim();
    if value.len() >= 2 {
        if let Some(stripped) = value.strip_prefix('"').and_then(|v| v.strip_suffix('"')) {
            return stripped.to_string();
        }
        if let Some(stripped) = value.strip_prefix('\'').and_then(|v| v.strip_suffix('\'')) {
            return stripped.to_string();
        }
    }
    value.to_string()
}

fn validate_skill_name(path: &Path, root: &Path, name: &str) -> Result<(), SkillLoadError> {
    if name.chars().count() > 64 {
        return Err(invalid_name(path, "must be 64 characters or fewer"));
    }
    if name.starts_with('-') || name.ends_with('-') {
        return Err(invalid_name(path, "must not start or end with a hyphen"));
    }
    if name.contains("--") {
        return Err(invalid_name(path, "must not contain consecutive hyphens"));
    }
    if !name
        .chars()
        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
    {
        return Err(invalid_name(
            path,
            "must contain only lowercase ASCII letters, numbers, and hyphens",
        ));
    }

    let dir_name = root
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();

    if dir_name != name {
        return Err(invalid_name(
            path,
            "Skill name must match parent directory name",
        ));
    }

    Ok(())
}

fn invalid_name(path: &Path, message: &str) -> SkillLoadError {
    SkillLoadError::InvalidField {
        path: path.to_path_buf(),
        field: "name",
        message: message.into(),
    }
}

fn discover_resources(root: &Path, skill_md: &Path) -> Result<Vec<SkillResource>, SkillLoadError> {
    let mut resources = Vec::new();
    discover_resources_inner(root, root, skill_md, &mut resources)?;
    resources.sort_by(|a, b| a.relative_path.cmp(&b.relative_path));
    Ok(resources)
}

fn discover_resources_inner(
    root: &Path,
    current: &Path,
    skill_md: &Path,
    resources: &mut Vec<SkillResource>,
) -> Result<(), SkillLoadError> {
    for entry in fs::read_dir(current)? {
        let entry = entry?;
        let path = entry.path();
        if is_hidden_path(&path) || path == skill_md {
            continue;
        }

        if path.is_dir() {
            discover_resources_inner(root, &path, skill_md, resources)?;
            continue;
        }

        if path.is_file() {
            let relative_path = path
                .strip_prefix(root)
                .map_err(|e| SkillLoadError::InvalidPath(e.to_string()))?
                .to_path_buf();
            let size_bytes = entry.metadata()?.len();
            let kind = classify_resource(&relative_path);
            resources.push(SkillResource {
                relative_path,
                kind,
                size_bytes,
            });
        }
    }

    Ok(())
}

fn classify_resource(path: &Path) -> SkillResourceKind {
    let first_component = path
        .components()
        .next()
        .and_then(|component| component.as_os_str().to_str());
    match first_component {
        Some("scripts") => return SkillResourceKind::Script,
        Some("references") => return SkillResourceKind::Reference,
        Some("assets") => return SkillResourceKind::Asset,
        _ => {}
    }

    match path.extension().and_then(|ext| ext.to_str()) {
        Some("md") | Some("markdown") => SkillResourceKind::Markdown,
        Some("txt") | Some("json") | Some("toml") | Some("yaml") | Some("yml") => {
            SkillResourceKind::Text
        }
        _ => SkillResourceKind::Other,
    }
}

fn is_hidden_path(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.starts_with('.'))
}
