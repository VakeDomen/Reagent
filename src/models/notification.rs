use serde::{Serialize};

use crate::services::ollama::models::{chat::{ChatRequest, ChatResponse}, tool::ToolCall};

pub type Success = bool;

#[derive(Debug, Clone, Serialize)]
pub struct Notification {
    pub agent: String,
    pub content: NotificationContent,
}


#[derive(Debug, Clone, Serialize)]
pub enum NotificationContent {
    Done(Success),
    PromptRequest(ChatRequest),
    PromptSuccessResult(ChatResponse),
    PromptErrorResult(String),
    ToolCallRequest(ToolCall),
    ToolCallSuccessResult(String),
    ToolCallErrorResult(String),
    McpToolNotification(String),
}