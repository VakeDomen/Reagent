---
name: bash
description: Use this skill when the user asks to inspect files, run shell commands, execute tests, list directories, check project state, or perform small local development tasks.
allowed_tools: bash
---

# Bash Skill

Use this skill when the task requires local shell access, such as:

- Listing files or directories
- Reading project structure
- Running tests
- Checking compiler errors
- Inspecting command output
- Performing small development diagnostics

## Rules

Before running a command:

1. Prefer read-only commands unless the user clearly asks for changes.
2. Keep commands focused and minimal.
3. Avoid destructive commands.
4. Do not run commands that remove files, overwrite data, install packages, change permissions, or access secrets unless the user explicitly requests it.
5. Do not expose environment variables, tokens, private keys, or credentials.

## Recommended Commands

For inspection:

```bash
pwd
ls
find . -maxdepth 3 -type f
cat <file>
sed -n '1,160p' <file>
grep -R "pattern" .
