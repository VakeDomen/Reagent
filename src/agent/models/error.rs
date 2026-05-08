use crate::{
    services::{llm::models::errors::InferenceClientError, mcp::error::McpIntegrationError},
    skills::SkillLoadError,
    templates::LoadTemplateError,
    InvocationError, ToolBuilderError, ToolExecutionError,
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
    /// A runtime failure, such as missing data or unexpected state.
    Runtime(String),
    /// A tool execution error, local or remote.
    Tool(ToolExecutionError),
    /// Failure when deserializing structured model output.
    Deserialization(serde_json::Error),
    /// Attempted to use a feature not yet supported by the provider or client.
    Unsupported(String),
    /// Invocation error, usually while building request shape during invocation.
    InvocationError(InvocationError),
}

impl std::fmt::Display for AgentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentError::InferenceClient(e) => write!(f, "Inference client error: {e}"),
            AgentError::AgentBuild(e) => write!(f, "Agent build error: {e}"),
            AgentError::Mcp(e) => write!(f, "MCP error: {e}"),
            AgentError::Runtime(s) => write!(f, "Runtime error: {s}"),
            AgentError::Tool(e) => write!(f, "Tool error: {e}"),
            AgentError::Deserialization(e) => write!(f, "Deserialization error: {e}"),
            AgentError::Unsupported(e) => write!(f, "Unsupported: {e}"),
            AgentError::InvocationError(e) => write!(f, "Invocation error: {e}"),
        }
    }
}

impl std::error::Error for AgentError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            AgentError::InferenceClient(e) => Some(e),
            AgentError::AgentBuild(e) => Some(e),
            AgentError::Mcp(e) => Some(e),
            AgentError::Runtime(_) => None,
            AgentError::Tool(e) => Some(e),
            AgentError::Deserialization(e) => Some(e),
            AgentError::Unsupported(_) => None,
            AgentError::InvocationError(e) => Some(e),
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
    /// A configured tool name conflicts with a reserved prebuilt tool.
    ReservedToolName(String),
    /// Failure while building a local tool definition.
    ToolBuild(ToolBuilderError),
    /// Failure while loading skill metadata or instructions.
    Skill(SkillLoadError),
    /// Failure while loading a prompt template from disk.
    TemplateLoad(LoadTemplateError),
}

impl std::fmt::Display for AgentBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentBuildError::InvalidJsonSchema(e) => {
                write!(f, "Invalid JSON schema provided: {e}")
            }
            AgentBuildError::McpError(e) => write!(f, "MCP error: {e}"),
            AgentBuildError::InferenceClient(e) => write!(f, "Inference client error: {e}"),
            AgentBuildError::ModelNotSet => write!(f, "Model not set."),
            AgentBuildError::Unsupported(e) => write!(f, "Unsupported: {e}"),
            AgentBuildError::ReservedToolName(name) => {
                write!(f, "Tool name `{name}` is reserved")
            }
            AgentBuildError::ToolBuild(e) => write!(f, "Tool build error: {e}"),
            AgentBuildError::Skill(e) => write!(f, "Skill error: {e}"),
            AgentBuildError::TemplateLoad(e) => write!(f, "Template load error: {e}"),
        }
    }
}

impl std::error::Error for AgentBuildError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            AgentBuildError::InvalidJsonSchema(_) => None,
            AgentBuildError::McpError(e) => Some(e),
            AgentBuildError::InferenceClient(e) => Some(e),
            AgentBuildError::ModelNotSet => None,
            AgentBuildError::Unsupported(_) => None,
            AgentBuildError::ReservedToolName(_) => None,
            AgentBuildError::ToolBuild(e) => Some(e),
            AgentBuildError::Skill(e) => Some(e),
            AgentBuildError::TemplateLoad(e) => Some(e),
        }
    }
}

impl From<InferenceClientError> for AgentBuildError {
    fn from(err: InferenceClientError) -> Self {
        AgentBuildError::InferenceClient(err)
    }
}

impl From<McpIntegrationError> for AgentBuildError {
    fn from(err: McpIntegrationError) -> Self {
        AgentBuildError::McpError(err)
    }
}

impl From<SkillLoadError> for AgentBuildError {
    fn from(err: SkillLoadError) -> Self {
        AgentBuildError::Skill(err)
    }
}

impl From<ToolBuilderError> for AgentBuildError {
    fn from(err: ToolBuilderError) -> Self {
        AgentBuildError::ToolBuild(err)
    }
}

impl From<LoadTemplateError> for AgentBuildError {
    fn from(err: LoadTemplateError) -> Self {
        AgentBuildError::TemplateLoad(err)
    }
}
