use serde::{Deserialize, Serialize};

use crate::{
    services::llm::{message::Message, models::base::BaseRequest},
    InvocationError, Tool,
};

#[derive(Serialize, Debug, Clone, Deserialize)]
pub struct ChatRequest {
    #[serde(flatten)]
    pub base: BaseRequest,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(bound(serialize = "T: Serialize", deserialize = "T: Deserialize<'de>"))]
pub struct ChatResponse<T = String> {
    pub model: String,
    pub created_at: String,
    pub message: Message<T>,
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

impl ChatResponse<String> {
    pub(crate) fn parse_content<T>(self) -> Result<ChatResponse<T>, InvocationError>
    where
        T: serde::de::DeserializeOwned,
    {
        let content = self.message.content.ok_or_else(|| {
            InvocationError::InvalidStructuredOutput("response had no content".into())
        })?;

        let content = serde_json::from_str(&content)
            .map_err(|error| InvocationError::InvalidStructuredOutput(error.to_string()))?;

        Ok(ChatResponse {
            model: self.model,
            created_at: self.created_at,
            message: Message {
                id: self.message.id,
                role: self.message.role,
                content: Some(content),
                thinking: self.message.thinking,
                images: self.message.images,
                tool_calls: self.message.tool_calls,
                tool_call_id: self.message.tool_call_id,
            },
            done: self.done,
            done_reason: self.done_reason,
            total_duration: self.total_duration,
            load_duration: self.load_duration,
            prompt_eval_count: self.prompt_eval_count,
            prompt_eval_duration: self.prompt_eval_duration,
            eval_count: self.eval_count,
            eval_duration: self.eval_duration,
        })
    }
}
