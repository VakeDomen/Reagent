use serde::{Serialize};

use crate::services::ollama::models::{chat::{ChatRequest, ChatResponse}, tool::ToolCall};

pub type Success = bool;

#[derive(Debug, Clone, Serialize)]
pub enum Notification {
    Done(Success),
    PromptRequest(ChatRequest),
    PromptSuccessResult(ChatResponse),
    PromptErrorResult(String),
    ToolCallRequest(ToolCall),
    ToolCallSuccessResult(String),
    ToolCallErrorResult(String),
}