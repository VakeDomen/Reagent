use crate::services::ollama::models::OllamaError;


#[derive(Debug)]
pub enum AgentError {
    OllamaError(OllamaError),
    // Add other potential agent-specific errors here
}

// Implement Display and Error for AgentError
impl std::fmt::Display for AgentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentError::OllamaError(e) => write!(f, "Ollama API Error: {}", e),
        }
    }
}

impl std::error::Error for AgentError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            AgentError::OllamaError(e) => Some(e),
        }
    }
}

// Implement From<OllamaError> for easy conversion with `?`
impl From<OllamaError> for AgentError {
    fn from(err: OllamaError) -> Self {
        AgentError::OllamaError(err)
    }
}
