use serde::{Deserialize, Serialize};

use crate::{services::llm::models::base::BaseRequest, Message, Tool};

#[derive(Serialize, Debug, Clone, Deserialize)]
pub struct ChatRequest {
    #[serde(flatten)]
    pub base: BaseRequest,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct ChatResponse {
    pub model: String,
    pub created_at: String,
    pub message: Message,
    pub done: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub done_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_duration: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub load_duration: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_eval_count: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_eval_duration: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eval_count: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eval_duration: Option<u64>,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct ChatStreamChunk {
    pub model: String,
    pub created_at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<Message>,
    pub done: bool,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub done_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_duration: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub load_duration: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_eval_count: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_eval_duration: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eval_count: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eval_duration: Option<u64>,
}
