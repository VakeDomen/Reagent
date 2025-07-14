use std::fmt::write;

use crate::services::{mcp::error::McpIntegrationError, ollama::models::errors::OllamaError};



#[derive(Debug)]
pub enum AgentError {
    OllamaError(OllamaError),
    AgentBuildError(AgentBuildError)
    // Add other potential agent-specific errors here
}

// Implement Display and Error for AgentError
impl std::fmt::Display for AgentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentError::OllamaError(e) => write!(f, "Ollama API Error: {}", e),
            AgentError::AgentBuildError(agent_build_error) => write!(f, "Agent Build Error: {}", agent_build_error),
                    }
    }
}

impl std::error::Error for AgentError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            AgentError::OllamaError(e) => Some(e),
            AgentError::AgentBuildError(agent_build_error) => Some(agent_build_error),
                    }
    }
}

// Implement From<OllamaError> for easy conversion with `?`
impl From<OllamaError> for AgentError {
    fn from(err: OllamaError) -> Self {
        AgentError::OllamaError(err)
    }
}


// Define a specific error type for builder configuration if parsing fails
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