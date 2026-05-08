pub static SKILL_SYSTEM_PROMPT_TEMPLATE: &str = r"
{{system_prompt}}

# Available Skills

You have access to specialized skills.

Each skill summary below is only for discovery. It is not the full instruction set.

When the user's task matches an available skill:
1. Call `read_skill` with the exact skill name.
2. Read the full skill instructions.
3. Apply those instructions to the user's task.

Do not perform a matching specialized task directly before loading the relevant skill.

If no skill matches the user's task, continue normally.

{{skills_discovery}}
";
