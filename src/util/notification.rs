use serde::{Deserialize, Serialize};

use crate::{services::ollama::models::chat::{ChatRequest, ChatResponse}, ToolCall};

pub type Success = bool;
pub type Response = Option<String>;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Token {
    pub tag: Option<String>,
    pub value: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Notification {
    pub agent: String,
    pub content: NotificationContent,
    pub mcp_envelope: Option<McpEnvelope>,
}


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


impl Notification {
    pub fn unwrap(self) -> Self {
        if let NotificationContent::McpToolNotification(ref mcp_string) = self.content {
        if let Ok(raw) = serde_json::from_str::<McpRaw>(mcp_string) {
            if let Ok(mut nested_notification) = serde_json::from_str::<Notification>(&raw.message) {
                nested_notification.mcp_envelope = Some(McpEnvelope { 
                    progress_token: raw.progress_token, 
                    progress: raw.progress 
                });
                return nested_notification.unwrap();
            }
        }
    }

    self
    }
}