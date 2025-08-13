#[derive(Debug)]
pub enum ToolExecutionError {
    ArgumentParsingError(String),
    ExecutionFailed(String),
    ToolNotFound(String),
}

impl std::fmt::Display for ToolExecutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ToolExecutionError::ArgumentParsingError(s) => write!(f, "Tool argument parsing error: {s}"),
            ToolExecutionError::ExecutionFailed(s) => write!(f, "Tool execution failed: {s}"),
            ToolExecutionError::ToolNotFound(s) => write!(f, "Tool not found: {s}"),
        }
    }
}

impl std::error::Error for ToolExecutionError {}