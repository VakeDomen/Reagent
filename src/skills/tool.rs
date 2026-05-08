use std::sync::Arc;

use serde_json::Value;

use super::{Skill, SkillResourceKind};
use crate::{AsyncToolFn, Tool, ToolBuilder, ToolBuilderError, ToolExecutionError};

pub fn build_read_skill_tool(skills: &[Skill]) -> Result<Tool, ToolBuilderError> {
    let skills = Arc::new(skills.to_vec());
    let executor: AsyncToolFn = Arc::new(move |args: Value| {
        let skills = Arc::clone(&skills);
        Box::pin(async move {
            let name = args.get("name").and_then(Value::as_str).ok_or_else(|| {
                ToolExecutionError::ArgumentParsingError(
                    "read_skill requires a string `name` argument".into(),
                )
            })?;

            let skill = skills
                .iter()
                .find(|skill| skill.name == name)
                .ok_or_else(|| ToolExecutionError::ToolNotFound(name.to_string()))?;

            Ok(format_skill_response(skill))
        })
    });

    ToolBuilder::new()
        .function_name("read_skill")
        .function_description(
            "Loads the full instructions for an available skill by exact skill name. \
            Call this before performing any user task that matches one of the available skills. \
            Use the returned instructions to complete the task.",
        )
        .add_required_property("name", "string", "Skill name to load")
        .executor(executor)
        .build()
}

fn format_skill_response(skill: &Skill) -> String {
    let mut response = String::new();
    response.push_str("# Skill: ");
    response.push_str(&skill.name);
    response.push_str("\n\n");
    response.push_str(&skill.raw_content);

    if !skill.resources.is_empty() {
        response.push_str("\n\n# Supporting Files\n\n");
        for resource in &skill.resources {
            response.push_str("- ");
            response.push_str(&resource.relative_path.display().to_string());
            response.push_str(" (");
            response.push_str(match &resource.kind {
                SkillResourceKind::Script => "script",
                SkillResourceKind::Reference => "reference",
                SkillResourceKind::Asset => "asset",
                SkillResourceKind::Markdown => "markdown",
                SkillResourceKind::Text => "text",
                SkillResourceKind::Other => "other",
            });
            response.push_str(", ");
            response.push_str(&resource.size_bytes.to_string());
            response.push_str(" bytes)\n");
        }
    }

    response
}
