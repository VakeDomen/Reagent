use crate::{
    services::{llm::models::errors::InferenceClientError, mcp::error::McpIntegrationError},
    InvocationError, ToolExecutionError,
};

/// Errors that can occur while running an [`Agent`].
#[derive(Debug)]
pub enum AgentError {
    /// Failure inside the underlying LLM client.
    InferenceClient(InferenceClientError),
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
    Unsupported(String),
    /// Invocation error (building request shape during invocation)
    InvocationError(InvocationError),
}

impl std::fmt::Display for AgentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentError::InferenceClient(e) => write!(f, "InferenceClientError API Error: {e}"),
            AgentError::AgentBuild(agent_build_error) => {
                write!(f, "Agent Build Error: {agent_build_error}")
            }
            AgentError::Mcp(mcp_integration_error) => {
                write!(f, "Mcp Error: {mcp_integration_error}")
            }
            AgentError::Runtime(s) => write!(f, "Runtime error: {s}"),
            AgentError::Deserialization(error) => write!(f, "Deserialize error: {error}"),
            AgentError::Tool(error) => write!(f, "Tool error: {error}"),
            AgentError::Unsupported(error) => write!(f, "Unsuppored: {:#?}", error),
            AgentError::InvocationError(error) => write!(f, "InvocationError: {:#?}", error),
        }
    }
}

impl std::error::Error for AgentError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            AgentError::InferenceClient(e) => Some(e),
            AgentError::AgentBuild(agent_build_error) => Some(agent_build_error),
            AgentError::Mcp(mcp_integration_error) => Some(mcp_integration_error),
            AgentError::Runtime(_) => Some(self),
            AgentError::Deserialization(error) => Some(error),
            AgentError::Tool(tool_execution_error) => Some(tool_execution_error),
            AgentError::Unsupported(_) => Some(self),
            AgentError::InvocationError(error) => Some(error),
        }
    }
}

impl From<InferenceClientError> for AgentError {
    fn from(err: InferenceClientError) -> Self {
        AgentError::InferenceClient(err)
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

impl From<InvocationError> for AgentError {
    fn from(err: InvocationError) -> Self {
        AgentError::InvocationError(err)
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
    InferenceClient(InferenceClientError),
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
            AgentBuildError::InferenceClient(e) => write!(f, "InferenceClient error: {e}"),
            AgentBuildError::Unsupported(error) => write!(f, "Unsuppored: {:#?}", error),
        }
    }
}

impl From<InferenceClientError> for AgentBuildError {
    fn from(err: InferenceClientError) -> Self {
        AgentBuildError::InferenceClient(err)
    }
}

impl std::error::Error for AgentBuildError {}
