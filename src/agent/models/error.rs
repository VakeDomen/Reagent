use crate::{services::{mcp::error::McpIntegrationError, ollama::models::errors::OllamaError}, ToolExecutionError};

#[derive(Debug)]
pub enum AgentError {
    Ollama(OllamaError),
    AgentBuild(AgentBuildError),
    Mcp(McpIntegrationError),
    Runtime(String),
    Tool(ToolExecutionError),
    Deserialization(serde_json::Error)
}

// Implement Display and Error for AgentError
impl std::fmt::Display for AgentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentError::Ollama(e) => write!(f, "Ollama API Error: {e}"),
            AgentError::AgentBuild(agent_build_error) => write!(f, "Agent Build Error: {agent_build_error}"),
            AgentError::Mcp(mcp_integration_error) => write!(f, "Mcp Error: {mcp_integration_error}"),
            AgentError::Runtime(s) => write!(f, "Runtime error: {s}"),
            AgentError::Deserialization(error) => write!(f, "Deserialize error: {error}"),
            AgentError::Tool(error) => write!(f, "Tool error: {error}"),
                    }
    }
}

impl std::error::Error for AgentError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            AgentError::Ollama(e) => Some(e),
            AgentError::AgentBuild(agent_build_error) => Some(agent_build_error),
            AgentError::Mcp(mcp_integration_error) => Some(mcp_integration_error),
            AgentError::Runtime(_) => Some(self),
            AgentError::Deserialization(error) => Some(error),
            AgentError::Tool(tool_execution_error) => Some(tool_execution_error),
                    }
    }
}

impl From<OllamaError> for AgentError {
    fn from(err: OllamaError) -> Self {
        AgentError::Ollama(err)
    }
}

impl From<AgentBuildError> for AgentError {
    fn from(err: AgentBuildError) -> Self {
        AgentError::AgentBuild(err)
    }
}

impl From<McpIntegrationError> for AgentError {
    fn from(err: McpIntegrationError) -> Self {
        AgentError::Mcp(err)
    }
}

impl From<ToolExecutionError> for AgentError {
    fn from(err: ToolExecutionError) -> Self {
        AgentError::Tool(err)
    }
}


#[derive(Debug)]
pub enum AgentBuildError {
    InvalidJsonSchema(String),
    McpError(McpIntegrationError),
    ModelNotSet
}

impl std::fmt::Display for AgentBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentBuildError::InvalidJsonSchema(e) => write!(f, "Invalid JSON schema provided: {e}"),
            AgentBuildError::ModelNotSet => write!(f, "Model not set."),
            AgentBuildError::McpError(e) => write!(f, "Mcp error: {e}"),
        }
    }
}
impl std::error::Error for AgentBuildError {}
