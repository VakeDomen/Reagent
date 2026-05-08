mod error;
mod loader;
mod models;
mod prompt;
mod tool;

pub use error::SkillLoadError;
pub use loader::{load_skill, load_skill_collection, load_skill_sources};
pub use models::{Skill, SkillResource, SkillResourceKind};
pub use tool::build_read_skill_tool;

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs,
        path::{Path, PathBuf},
        time::{SystemTime, UNIX_EPOCH},
    };

    fn temp_skill_root(name: &str) -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("reagent-skill-test-{name}-{stamp}"))
    }

    fn write_skill(root: &Path, name: &str, description: &str, body: &str) {
        fs::create_dir_all(root).unwrap();
        fs::write(
            root.join("SKILL.md"),
            format!(
                "---\nname: {name}\ndescription: {description}\nmetadata:\n  author: reagent\n---\n{body}"
            ),
        )
        .unwrap();
    }

    #[test]
    fn loads_skill_from_directory() {
        let root = temp_skill_root("single");
        let skill_root = root.join("pdf-processing");
        write_skill(
            &skill_root,
            "pdf-processing",
            "Extract PDF content. Use when working with PDFs.",
            "Read PDFs carefully.",
        );
        fs::create_dir_all(skill_root.join("references")).unwrap();
        fs::write(
            skill_root.join("references").join("REFERENCE.md"),
            "details",
        )
        .unwrap();

        let skill = load_skill(&skill_root).unwrap();
        assert_eq!(skill.name, "pdf-processing");
        assert_eq!(skill.metadata.get("author").unwrap(), "reagent");
        assert!(skill.raw_content.contains("Read PDFs carefully."));
        assert_eq!(skill.resources.len(), 1);

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn loads_skill_collection_in_sorted_order() {
        let root = temp_skill_root("collection");
        write_skill(
            &root.join("zeta-skill"),
            "zeta-skill",
            "Zeta skill. Use for zeta tasks.",
            "Zeta instructions.",
        );
        write_skill(
            &root.join("alpha-skill"),
            "alpha-skill",
            "Alpha skill. Use for alpha tasks.",
            "Alpha instructions.",
        );

        let skills = load_skill_collection(&root).unwrap();
        assert_eq!(skills[0].name, "alpha-skill");
        assert_eq!(skills[1].name, "zeta-skill");

        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn read_skill_tool_returns_full_skill_file() {
        let root = temp_skill_root("tool");
        let skill_root = root.join("data-cleaning");
        write_skill(
            &skill_root,
            "data-cleaning",
            "Clean data. Use when processing messy data.",
            "Normalize columns.",
        );
        let skill = load_skill(&skill_root).unwrap();
        let tool = build_read_skill_tool(&[skill]).unwrap();

        let output = tool
            .execute(serde_json::json!({ "name": "data-cleaning" }))
            .await
            .unwrap();
        assert!(output.contains("name: data-cleaning"));
        assert!(output.contains("Normalize columns."));

        fs::remove_dir_all(root).unwrap();
    }
}
