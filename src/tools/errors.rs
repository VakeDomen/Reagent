/// Errors that can occur during execution of a tool.
///
/// These errors indicate failures in parsing arguments, actually
/// running the tool, or locating the requested tool.
#[derive(Debug)]
pub enum ToolExecutionError {
    /// The provided arguments could not be parsed or were invalid.
    ArgumentParsingError(String),
    /// The tool failed during execution (runtime failure inside the tool).
    ExecutionFailed(String),
    /// The requested tool was not found in the agent’s registry.
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