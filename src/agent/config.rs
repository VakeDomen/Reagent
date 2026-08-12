use crate::{services::llm::SchemaSpec, templates::Template, McpServerType, Tool};

#[derive(Debug, Clone, Default)]
pub struct PromptConfig {
    /// Optional prompt template used to compile user inputs.
    pub template: Option<Template>,
    /// System prompt that seeds the conversation.
    pub system_prompt: Option<String>,
    /// Set of local tools the agent can invoke.
    pub tools: Option<Vec<Tool>>,
    /// The normalized, typed form used by Agent and provider adapters
    pub response_format: Option<SchemaSpec>,
    /// Optional raw JSON string the user gave; parsed and merged at build
    pub response_format_raw: Option<String>,
    /// Optional hint when caller set only a raw string
    pub pending_name: Option<String>,
    /// Optional hint when caller set only a raw string
    pub pending_strict: Option<bool>,
    /// External MCP servers providing additional tools.
    pub mcp_servers: Option<Vec<McpServerType>>,
    /// Prompt injected at the start of tool-call branches.
    pub stop_prompt: Option<String>,
    /// Stopword used to detect end of model output.
    pub stopword: Option<String>,
    /// Safety cap on maximum number of conversation iterations.
    pub max_iterations: Option<usize>,
    /// Whether to clear conversation history before each invocation.
    pub clear_histroy_on_invoke: Option<bool>,
    /// Enable streaming responses (token-by-token).
    pub stream: bool,
}
