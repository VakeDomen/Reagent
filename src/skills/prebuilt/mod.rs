use std::{collections::HashMap, path::PathBuf};

use crate::skills::{Skill, SkillResource};

pub static BASH_SKILL_MD: &str = include_str!("bash_skill/SKILL.md");

pub fn bash_skill() -> Skill {
    Skill {
        name: "bash".to_string(),
        description: "Use this skill when the user asks to inspect files, run shell commands, execute tests, list directories, check project state, or perform small local development tasks.".to_string(),
        root: PathBuf::from("<builtin>/bash"),
        skill_md: PathBuf::from("<builtin>/bash/SKILL.md"),
        raw_content: BASH_SKILL_MD.to_string(),
        instructions: extract_instructions(BASH_SKILL_MD),
        license: None,
        compatibility: None,
        metadata: HashMap::new(),
        allowed_tools: Some("bash".to_string()),
        resources: Vec::<SkillResource>::new(),
    }
}

fn extract_instructions(raw: &str) -> String {
    if raw.trim_start().starts_with("---") {
        let mut parts = raw.splitn(3, "---");
        let _ = parts.next();
        let _frontmatter = parts.next();
        parts.next().unwrap_or(raw).trim().to_string()
    } else {
        raw.trim().to_string()
    }
}
