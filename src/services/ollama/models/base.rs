use std::collections::HashMap;

use serde::{Deserialize, Serialize};
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
    pub options: Option<HashMap<String, serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<String>, // e.g., "5m"
}

