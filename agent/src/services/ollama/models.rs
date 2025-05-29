use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug)]
pub enum OllamaError {
    RequestError(String),
    ApiError(String),
    SerializationError(String),
}

impl std::fmt::Display for OllamaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OllamaError::RequestError(s) => write!(f, "Request Error: {}", s),
            OllamaError::ApiError(s) => write!(f, "API Error: {}", s),
            OllamaError::SerializationError(s) => write!(f, "Serialization Error: {}", s),
        }
    }
}

impl std::error::Error for OllamaError {}

/// Represents the role of a message sender in a chat.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Message {
    pub role: Role,
    // Make content optional and provide a default if it's missing in the JSON
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")] // Ensure this is also default
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl Message {
    /// Creates a new message with a specific role and content.
    pub fn new(role: Role, content: String) -> Self {
        Self {
            role,
            content: Some(content),
            images: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Creates a new 'system' message.
    pub fn system(content: String) -> Self {
        Self::new(Role::System, content)
    }

    /// Creates a new 'user' message.
    pub fn user(content: String) -> Self {
        Self::new(Role::User, content)
    }

    /// Creates a new 'assistant' message.
    pub fn assistant(content: String) -> Self {
        Self::new(Role::Assistant, content)
    }

    pub fn tool(content: String, tool_call_id: String) -> Self {
        Self {
            role: Role::Tool,
            content: Some(content),
            images: None,
            tool_calls: None,
            tool_call_id: Some(tool_call_id),
        }
    }
}
/// Base structure for requests.
#[derive(Serialize, Debug, Clone, Default)]
pub struct BaseRequest {
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<String>, // e.g., "json"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<HashMap<String, serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<String>, // e.g., "5m"
}

/// Request for the `/api/generate` endpoint.
#[derive(Serialize, Debug, Clone)]
pub struct GenerateRequest {
    #[serde(flatten)]
    pub base: BaseRequest,
    pub prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub template: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<Vec<i64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw: Option<bool>,
}

/// Response from the `/api/generate` endpoint.
///
/// This structure represents a single response object. If streaming is disabled,
/// it contains the full response. If streaming is enabled, multiple `GenerateResponse`
/// objects will be received, with the final one containing the performance statistics.
#[derive(Deserialize, Debug, Clone)]
pub struct GenerateResponse {
    /// The model name used for generation.
    pub model: String,
    /// The timestamp when the response was created.
    pub created_at: String,
    /// The generated response content. This will be an aggregation if `stream` is false.
    pub response: String,
    /// Indicates if this is the final response (`true`) or part of a stream (`false`).
    pub done: bool,
    /// A reason for why the generation finished. This is only present when `done` is `true`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub done_reason: Option<String>,
    /// An encoding of the conversation context. This can be sent in the next request
    /// to maintain conversational memory. Present only if `done` is `true`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<Vec<i64>>,
    /// Time spent generating the response (nanoseconds). Present only if `done` is `true`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_duration: Option<u64>,
    /// Time spent loading the model (nanoseconds). Present only if `done` is `true`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub load_duration: Option<u64>,
    /// Number of tokens in the prompt. Present only if `done` is `true`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_eval_count: Option<u32>,
    /// Time spent evaluating the prompt (nanoseconds). Present only if `done` is `true`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_eval_duration: Option<u64>,
    /// Number of tokens in the response. Present only if `done` is `true`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eval_count: Option<u32>,
    /// Time spent generating the response (nanoseconds). Present only if `done` is `true`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eval_duration: Option<u64>,
}


/// Request for the `/api/chat` endpoint.
#[derive(Serialize, Debug, Clone)]
pub struct ChatRequest {
    #[serde(flatten)]
    pub base: BaseRequest,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
}


/// Response from the `/api/chat` endpoint.
///
/// This structure represents a single chat response object. If streaming is disabled,
/// it contains the full message. If streaming is enabled, multiple `ChatResponse`
/// objects will be received (each containing a chunk of the message), with the
/// final one containing the performance statistics.
#[derive(Deserialize, Debug, Clone)]
pub struct ChatResponse {
    /// The model name used for the chat.
    pub model: String,
    /// The timestamp when the response was created.
    pub created_at: String,
    /// The message generated by the model. This might be a partial message if streaming.
    pub message: Message,
    /// Indicates if this is the final response (`true`) or part of a stream (`false`).
    pub done: bool,
    /// A reason for why the generation finished. This is only present when `done` is `true`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub done_reason: Option<String>,
    /// Time spent generating the response (nanoseconds). Present only if `done` is `true`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_duration: Option<u64>,
    /// Time spent loading the model (nanoseconds). Present only if `done` is `true`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub load_duration: Option<u64>,
    /// Number of tokens in the prompt. Present only if `done` is `true`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_eval_count: Option<u32>,
    /// Time spent evaluating the prompt (nanoseconds). Present only if `done` is `true`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_eval_duration: Option<u64>,
    /// Number of tokens in the response. Present only if `done` is `true`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eval_count: Option<u32>,
    /// Time spent generating the response (nanoseconds). Present only if `done` is `true`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eval_duration: Option<u64>,
}


#[derive(Serialize, Debug, Clone)]
pub struct EmbeddingsRequest {
    pub model: String,
    pub input: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<HashMap<String, serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<String>,
}

/// Response from the `/api/embeddings` endpoint.
#[derive(Deserialize, Debug, Clone)]
pub struct EmbeddingsResponse {
    pub embedding: Vec<f64>, // Ollama typically returns f64
}


/// Defines the type of tool available. Currently, only 'function' is supported.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ToolType {
    Function,
}

/// Defines a tool (function) that the model can call.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tool_type: ToolType,
    pub function: Function,
}

/// Defines a function, its description, and its parameters.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Function {
    pub name: String,
    pub description: String,
    pub parameters: FunctionParameters,
}

/// Defines the parameters for a function using a JSON schema-like structure.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FunctionParameters {
    #[serde(rename = "type")]
    pub param_type: String, // Typically "object"
    pub properties: HashMap<String, Property>,
    pub required: Vec<String>,
}

/// Defines a single property within function parameters.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Property {
    #[serde(rename = "type")]
    pub property_type: String, // e.g., "string", "number", "integer"
    pub description: String,
}

/// Represents a tool call requested by the model.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ToolCall {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default = "default_tool_call_type", skip_serializing_if = "is_default_tool_call_type")]
    #[serde(rename = "type")]
    pub tool_type: ToolType, // Should always be "function" for now
    pub function: ToolCallFunction,
}

// Helper function to provide a default ToolType if it's missing in the JSON
// This is used if your Ollama version consistently omits the "type" field in the tool_call object.
fn default_tool_call_type() -> ToolType {
    ToolType::Function // Default to function
}

// Helper for skip_serializing_if to avoid serializing if it's the default
// This is more relevant if you were to serialize this struct often and wanted to omit default values.
#[allow(clippy::trivially_copy_pass_by_ref)] // Added to address potential clippy lint
fn is_default_tool_call_type(tool_type: &ToolType) -> bool {
    *tool_type == default_tool_call_type()
}

/// Contains the name and arguments for a function call.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ToolCallFunction {
    pub name: String,
    pub arguments: serde_json::Value,
}