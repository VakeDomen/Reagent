use serde::{Deserialize, Serialize};

use crate::{services::ollama::models::chat::{ChatRequest, ChatResponse}, ToolCall};


#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationContent {
    Done(Success, Response),
    PromptRequest(ChatRequest),
    PromptSuccessResult(ChatResponse),
    PromptErrorResult(String),
    ToolCallRequest(ToolCall),
    ToolCallSuccessResult(String),
    ToolCallErrorResult(String),
    Token(Token),
    McpToolNotification(String),
}


pub type Success = bool;
pub type Response = Option<String>;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Token {
    pub tag: Option<String>,
    pub value: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpEnvelope {
    pub progress_token: i32,
    pub progress: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpRaw {
    pub progress_token: i32,
    pub progress: i32,
    pub message: String,
}
