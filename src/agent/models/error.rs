use crate::{services::{mcp::error::McpIntegrationError, llm::models::errors::ModelClientError}, ToolExecutionError};

/// Errors that can occur while running an [`Agent`].
#[derive(Debug)]
pub enum AgentError {
    /// Failure inside the underlying LLM client.
    ModelClient(ModelClientError),
    /// Errors that occur during agent construction.
    AgentBuild(AgentBuildError),
    /// Integration errors when connecting to MCP servers.
    Mcp(McpIntegrationError),
    /// A runtime failure (e.g. missing data, unexpected state).
    Runtime(String),
    /// A tool execution error (local or remote).
    Tool(ToolExecutionError),
    /// Failure when deserializing structured model output.
    Deserialization(serde_json::Error),
    /// Attempted to use a feature not yet supported by the provider or client.
    Unsupported(String)
}

impl std::fmt::Display for AgentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentError::ModelClient(e) => write!(f, "ModelClientError API Error: {e}"),
            AgentError::AgentBuild(agent_build_error) => write!(f, "Agent Build Error: {agent_build_error}"),
            AgentError::Mcp(mcp_integration_error) => write!(f, "Mcp Error: {mcp_integration_error}"),
            AgentError::Runtime(s) => write!(f, "Runtime error: {s}"),
            AgentError::Deserialization(error) => write!(f, "Deserialize error: {error}"),
            AgentError::Tool(error) => write!(f, "Tool error: {error}"),
            AgentError::Unsupported(error) => write!(f, "Unsuppored: {:#?}", error),
        }
    }
}

impl std::error::Error for AgentError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            AgentError::ModelClient(e) => Some(e),
            AgentError::AgentBuild(agent_build_error) => Some(agent_build_error),
            AgentError::Mcp(mcp_integration_error) => Some(mcp_integration_error),
            AgentError::Runtime(_) => Some(self),
            AgentError::Deserialization(error) => Some(error),
            AgentError::Tool(tool_execution_error) => Some(tool_execution_error),
            AgentError::Unsupported(_) => Some(self)
        }
    }
}

impl From<ModelClientError> for AgentError {
    fn from(err: ModelClientError) -> Self {
        AgentError::ModelClient(err)
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


/// Errors that can occur while building an [`Agent`].
#[derive(Debug)]
pub enum AgentBuildError {
    /// Provided JSON schema for response format could not be parsed.
    InvalidJsonSchema(String),
    /// Failure while compiling tools from MCP integration.
    McpError(McpIntegrationError),
    /// Failure initializing the underlying model client.
    ModelClient(ModelClientError),
    /// Required model was not set on the builder.
    ModelNotSet,
    /// Attempted to use a feature not yet supported by the provider or client.
    Unsupported(String),
}

impl std::fmt::Display for AgentBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentBuildError::InvalidJsonSchema(e) => write!(f, "Invalid JSON schema provided: {e}"),
            AgentBuildError::ModelNotSet => write!(f, "Model not set."),
            AgentBuildError::McpError(e) => write!(f, "Mcp error: {e}"),
            AgentBuildError::ModelClient(e) => write!(f, "ModelClient error: {e}"),
            AgentBuildError::Unsupported(error) => write!(f, "Unsuppored: {:#?}", error),
        }
    }
}

impl From<ModelClientError> for AgentBuildError {
    fn from(err: ModelClientError) -> Self {
        AgentBuildError::ModelClient(err)
    }
}


impl std::error::Error for AgentBuildError {}
