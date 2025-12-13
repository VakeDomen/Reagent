use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{Role, ToolCall};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Message {
    #[serde(default = "new_uuid", skip_serializing)]
    pub id: String,
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
    fn new(role: Role, content: String, tool_call_id: Option<String>) -> Self {
        Self {
            id: new_uuid(),
            role,
            content: Some(content),
            images: None,
            tool_calls: None,
            tool_call_id,
        }
    }

    pub fn system<T: Into<String>>(content: T) -> Self {
        Self::new(Role::System, content.into(), None)
    }
    pub fn developer<T: Into<String>>(content: T) -> Self {
        Self::new(Role::Developer, content.into(), None)
    }
    pub fn user<T: Into<String>>(content: T) -> Self {
        Self::new(Role::User, content.into(), None)
    }
    pub fn assistant<T: Into<String>>(content: T) -> Self {
        Self::new(Role::Assistant, content.into(), None)
    }
    pub fn tool<T, S>(content: T, tool_call_id: S) -> Self
    where
        T: Into<String>,
        S: Into<String>,
    {
        Self::new(Role::Tool, content.into(), Some(tool_call_id.into()))
    }
}

fn new_uuid() -> String {
    Uuid::new_v4().to_string()
}
