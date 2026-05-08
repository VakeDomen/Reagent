use std::{fmt, path::PathBuf};

#[derive(Debug, Clone)]
pub enum SkillLoadError {
    Io(String),
    InvalidPath(String),
    MissingSkillFile(PathBuf),
    MissingFrontmatter(PathBuf),
    MissingField {
        path: PathBuf,
        field: &'static str,
    },
    InvalidField {
        path: PathBuf,
        field: &'static str,
        message: String,
    },
    DuplicateSkillName(String),
    NoSkillsInCollection(PathBuf),
}

impl fmt::Display for SkillLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SkillLoadError::Io(e) => write!(f, "Skill IO error: {e}"),
            SkillLoadError::InvalidPath(e) => write!(f, "Invalid skill path: {e}"),
            SkillLoadError::MissingSkillFile(path) => {
                write!(f, "No SKILL.md or skill.md found at {}", path.display())
            }
            SkillLoadError::MissingFrontmatter(path) => {
                write!(
                    f,
                    "Skill file is missing YAML frontmatter: {}",
                    path.display()
                )
            }
            SkillLoadError::MissingField { path, field } => {
                write!(
                    f,
                    "Skill file {} is missing required field `{field}`",
                    path.display()
                )
            }
            SkillLoadError::InvalidField {
                path,
                field,
                message,
            } => write!(
                f,
                "Skill file {} has invalid field `{field}`: {message}",
                path.display()
            ),
            SkillLoadError::DuplicateSkillName(name) => {
                write!(f, "Duplicate skill name `{name}`")
            }
            SkillLoadError::NoSkillsInCollection(path) => {
                write!(f, "No skills found in collection {}", path.display())
            }
        }
    }
}

impl std::error::Error for SkillLoadError {}

impl From<std::io::Error> for SkillLoadError {
    fn from(err: std::io::Error) -> Self {
        SkillLoadError::Io(err.to_string())
    }
}
