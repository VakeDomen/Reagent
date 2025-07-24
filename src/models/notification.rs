use serde::{Deserialize, Serialize};

use crate::services::ollama::models::{chat::{ChatRequest, ChatResponse}, tool::ToolCall};

pub type Success = bool;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Notification {
    pub agent: String,
    pub content: NotificationContent,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
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


impl Notification {
    pub fn unwrap(self) -> Self {
        if let NotificationContent::McpToolNotification(ref mcp_string) = self.content {
            if let Ok(nested_notification) = serde_json::from_str::<Notification>(mcp_string) {
                return nested_notification.unwrap();
            }
        }
        
        self
    }
}