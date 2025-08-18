use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::{
    notifications::notiifcation_content::{McpEnvelope, McpRaw}, 
    NotificationContent
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Notification {
    pub agent: String,
    pub content: NotificationContent,
    pub mcp_envelope: Option<McpEnvelope>,
    pub timestamp_millis: u128,
}


impl Notification {
    pub fn new(agent: String, content: NotificationContent) -> Self {
        Self { 
            agent, 
            content, 
            mcp_envelope: None, 
            timestamp_millis: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("Time should go forward")
                .as_millis()
        }
    }    

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