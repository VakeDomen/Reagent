use std::{collections::HashMap, path::PathBuf};

#[derive(Debug, Clone)]
pub struct Skill {
    pub name: String,
    pub description: String,
    pub root: PathBuf,
    pub skill_md: PathBuf,
    pub raw_content: String,
    pub instructions: String,
    pub license: Option<String>,
    pub compatibility: Option<String>,
    pub metadata: HashMap<String, String>,
    pub allowed_tools: Option<String>,
    pub resources: Vec<SkillResource>,
}

#[derive(Debug, Clone)]
pub struct SkillResource {
    pub relative_path: PathBuf,
    pub kind: SkillResourceKind,
    pub size_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SkillResourceKind {
    Script,
    Reference,
    Asset,
    Markdown,
    Text,
    Other,
}
