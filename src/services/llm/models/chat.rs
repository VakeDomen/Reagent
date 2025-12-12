use serde::{Deserialize, Serialize};

use crate::{
    services::llm::{message::Message, models::base::BaseRequest, InferenceOptions},
    Agent, Tool,
};

#[derive(Serialize, Debug, Clone, Deserialize)]
pub struct ChatRequest {
    #[serde(flatten)]
    pub base: BaseRequest,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
}

impl From<&Agent> for ChatRequest {
    fn from(val: &Agent) -> Self {
        let options = InferenceOptions {
            num_ctx: val.num_ctx,
            repeat_last_n: val.repeat_last_n,
            repeat_penalty: val.repeat_penalty,
            temperature: val.temperature,
            seed: val.seed,
            stop: val.stop.clone(),
            num_predict: val.num_predict,
            top_k: val.top_k,
            top_p: val.top_p,
            min_p: val.min_p,
            presence_penalty: val.presence_penalty,
            frequency_penalty: val.frequency_penalty,
            max_tokens: None,
        };
        ChatRequest {
            base: BaseRequest {
                model: val.model.clone(),
                format: val.response_format.clone(),
                options: Some(options),
                stream: Some(val.stream),
                keep_alive: Some("5m".to_string()),
            },
            messages: val.history.clone(),
            tools: val.tools.clone(),
        }
    }
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
