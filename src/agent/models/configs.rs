use crate::{services::llm::SchemaSpec, templates::Template, McpServerType, Tool};

#[derive(Debug, Clone, Default)]
pub struct ModelConfig {
    /// Identifier or name of the model to use.
    pub model: Option<String>,
    /// Sampling temperature (`0.0` = deterministic, higher = more random).
    pub temperature: Option<f32>,
    /// Nucleus sampling probability threshold.
    pub top_p: Option<f32>,
    /// Penalty applied to encourage novel tokens.
    pub presence_penalty: Option<f32>,
    /// Penalty applied to discourage repeating tokens.
    pub frequency_penalty: Option<f32>,
    /// Maximum context length (in tokens).
    pub num_ctx: Option<u32>,
    /// Number of tokens to look back when applying repeat penalty.
    pub repeat_last_n: Option<i32>,
    /// Strength of repeat penalty.
    pub repeat_penalty: Option<f32>,
    /// RNG seed for reproducibility.
    pub seed: Option<i32>,
    /// Hard stop sequence that forces termination.
    pub stop: Option<String>,
    /// Maximum number of tokens to generate in a single response.
    pub num_predict: Option<i32>,
    /// Top-K cutoff for sampling (how many candidates are considered).
    pub top_k: Option<u32>,
    /// Minimum probability threshold for token acceptance.
    pub min_p: Option<f32>,
}

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
    /// Whether to strip `<think>` blocks from model responses.
    pub strip_thinking: Option<bool>,
    /// Safety cap on maximum number of conversation iterations.
    pub max_iterations: Option<usize>,
    /// Whether to clear conversation history before each invocation.
    pub clear_histroy_on_invoke: Option<bool>,
    /// Enable streaming responses (token-by-token).
    pub stream: bool,
}
