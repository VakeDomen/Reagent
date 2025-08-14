use serde::{Deserialize, Serialize, Serializer};
use serde_json::Value;

use crate::ToolCall;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    Developer,
    User,
    Assistant,
    Tool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Message {
    pub role: Role,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl Message {
    pub fn new(role: Role, content: String) -> Self {
        Self { role, content: Some(content), images: None, tool_calls: None, tool_call_id: None }
    }

    pub fn system<T: Into<String>>(content: T) -> Self { Self::new(Role::System, content.into()) }
    pub fn developer<T: Into<String>>(content: T) -> Self { Self::new(Role::Developer, content.into()) }
    pub fn user<T: Into<String>>(content: T) -> Self { Self::new(Role::User, content.into()) }
    pub fn assistant<T: Into<String>>(content: T) -> Self { Self::new(Role::Assistant, content.into()) }
    pub fn tool<T, S>(content: T, tool_call_id: S) -> Self where T: Into<String>, S: Into<String> {
        Self { role: Role::Tool, content: Some(content.into()), images: None, tool_calls: None, tool_call_id: Some(tool_call_id.into()) }
    }
}

#[derive(Serialize, Debug, Clone, Default, Deserialize)]
pub struct BaseRequest {
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<Value>,

    #[serde(flatten)]
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "options")]
    #[serde(serialize_with = "serialize_options_as_map")]
    pub options: Option<InferenceOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct InferenceOptions {
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
    pub max_tokens: Option<i32>,
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

fn serialize_options_as_map<S>(options: &Option<InferenceOptions>, serializer: S) -> Result<S::Ok, S::Error>
where S: Serializer {
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