use crate::services::{mcp::error::McpIntegrationError, ollama::models::errors::OllamaError};

#[derive(Debug)]
pub enum AgentError {
    OllamaError(OllamaError),
    AgentBuildError(AgentBuildError),
    McpError(McpIntegrationError),
}

// Implement Display and Error for AgentError
impl std::fmt::Display for AgentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentError::OllamaError(e) => write!(f, "Ollama API Error: {}", e),
            AgentError::AgentBuildError(agent_build_error) => write!(f, "Agent Build Error: {}", agent_build_error),
            AgentError::McpError(mcp_integration_error) => write!(f, "Mcp Error: {}", mcp_integration_error),
        }
    }
}

impl std::error::Error for AgentError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            AgentError::OllamaError(e) => Some(e),
            AgentError::AgentBuildError(agent_build_error) => Some(agent_build_error),
            AgentError::McpError(mcp_integration_error) => Some(mcp_integration_error),
        }
    }
}

impl From<OllamaError> for AgentError {
    fn from(err: OllamaError) -> Self {
        AgentError::OllamaError(err)
    }
}

impl From<AgentBuildError> for AgentError {
    fn from(err: AgentBuildError) -> Self {
        AgentError::AgentBuildError(err)
    }
}

impl From<McpIntegrationError> for AgentError {
    fn from(err: McpIntegrationError) -> Self {
        AgentError::McpError(err)
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
            AgentBuildError::InvalidJsonSchema(e) => write!(f, "Invalid JSON schema provided: {}", e),
            AgentBuildError::ModelNotSet => write!(f, "Model not set."),
            AgentBuildError::McpError(e) => write!(f, "Mcp error: {}", e),
        }
    }
}
impl std::error::Error for AgentBuildError {}