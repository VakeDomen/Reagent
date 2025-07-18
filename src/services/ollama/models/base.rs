
use serde::{Deserialize, Serialize, Serializer};
use serde_json::Value;

use super::tool::ToolCall;

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
    pub fn system<T: Into<String>>(content: T) -> Self {
        Self::new(Role::System, content.into())
    }

    /// Creates a new 'user' message.
    pub fn user<T: Into<String>>(content: T) -> Self {
        Self::new(Role::User, content.into())
    }

    /// Creates a new 'assistant' message.
    pub fn assistant<T: Into<String>>(content: T) -> Self {
        Self::new(Role::Assistant, content.into())
    }

    pub fn tool<T, S>(content: T, tool_call_id: S) -> Self where T: Into<String>, S: Into<String> {
        Self {
            role: Role::Tool,
            content: Some(content.into()),
            images: None,
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
        }
    }
}
/// Base structure for requests.
#[derive(Serialize, Debug, Clone, Default)]
pub struct BaseRequest {
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "options")]
    #[serde(serialize_with = "serialize_options_as_map")]
    pub options: Option<OllamaOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<String>, // e.g., "5m"
}


/// Structured options for customizing Ollama request behavior.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OllamaOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_ctx: Option<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat_last_n: Option<i32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat_penalty: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_predict: Option<i32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
}

fn serialize_options_as_map<S>(
    options: &Option<OllamaOptions>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    match options {
        Some(opts) => {
            let map = serde_json::to_value(opts)
                .map_err(serde::ser::Error::custom)?
                .as_object()
                .cloned()
                .unwrap_or_default();

            map.serialize(serializer)
        }
        None => serializer.serialize_none(),
    }
}