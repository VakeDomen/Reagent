use std::{path::PathBuf, sync::Arc, time::Duration};

use serde_json::Value;
use tokio::{process::Command, time::timeout};

use crate::{AsyncToolFn, Tool, ToolBuilder, ToolBuilderError, ToolExecutionError};

#[derive(Debug, Clone)]
pub struct BashToolConfig {
    /// Working directory for commands.
    pub cwd: PathBuf,

    /// Timeout per command.
    pub timeout: Duration,

    /// Maximum number of stdout chars returned to the model.
    pub max_stdout_chars: usize,

    /// Maximum number of stderr chars returned to the model.
    pub max_stderr_chars: usize,
}

impl Default for BashToolConfig {
    fn default() -> Self {
        Self {
            cwd: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            timeout: Duration::from_secs(15),
            max_stdout_chars: 12_000,
            max_stderr_chars: 8_000,
        }
    }
}

pub fn build_bash_tool(config: BashToolConfig) -> Result<Tool, ToolBuilderError> {
    let config = Arc::new(config);

    let executor: AsyncToolFn = Arc::new(move |args: Value| {
        let config = Arc::clone(&config);

        Box::pin(async move {
            let command = args
                .get("command")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    ToolExecutionError::ArgumentParsingError(
                        "bash requires a non-empty string `command` argument".into(),
                    )
                })?;

            let output = timeout(
                config.timeout,
                Command::new("bash")
                    .arg("-lc")
                    .arg(command)
                    .current_dir(&config.cwd)
                    .output(),
            )
            .await;

            let output = match output {
                Ok(Ok(output)) => output,
                Ok(Err(err)) => {
                    return Ok(format!("Failed to execute bash command.\n\nError:\n{err}"));
                }
                Err(_) => {
                    return Ok(format!(
                        "Bash command timed out after {} seconds.",
                        config.timeout.as_secs()
                    ));
                }
            };

            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);

            let stdout = truncate_chars(&stdout, config.max_stdout_chars);
            let stderr = truncate_chars(&stderr, config.max_stderr_chars);

            let exit_code = output
                .status
                .code()
                .map(|code| code.to_string())
                .unwrap_or_else(|| "terminated by signal".to_string());

            Ok(format!(
                "Command:\n{command}\n\nExit code:\n{exit_code}\n\nStdout:\n{stdout}\n\nStderr:\n{stderr}"
            ))
        })
    });

    ToolBuilder::new()
        .function_name("bash")
        .function_description(
            "Executes a bash command in the configured working directory and returns stdout, stderr, and exit code. \
             Use this for local filesystem inspection, running tests, checking files, and small development tasks.",
        )
        .add_required_property(
            "command",
            "string",
            "The bash command to execute. Example: `ls -la` or `cargo test`.",
        )
        .executor(executor)
        .build()
}

fn truncate_chars(input: &str, max_chars: usize) -> String {
    let mut chars = input.chars();

    let truncated: String = chars.by_ref().take(max_chars).collect();

    if chars.next().is_some() {
        format!("{truncated}\n\n[output truncated]")
    } else {
        truncated
    }
}
