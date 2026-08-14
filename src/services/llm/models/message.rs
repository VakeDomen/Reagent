use serde::{de::Error as _, Deserialize, Deserializer, Serialize};
use serde_json::Value;
use uuid::Uuid;

use crate::{Role, ToolCall};

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(bound(
    serialize = "T: Serialize",
    deserialize = "T: serde::de::DeserializeOwned"
))]
pub struct Message<T = String> {
    #[serde(default = "new_uuid", skip_serializing)]
    pub id: String,
    pub role: Role,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_content"
    )]
    pub content: Option<T>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// Provider APIs encode assistant content as a string, including when that
/// string itself contains structured JSON. Decode both representations through
/// serde so `Message<T>` owns the output type.
fn deserialize_content<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: serde::de::DeserializeOwned,
{
    let content = Option::<Value>::deserialize(deserializer)?;
    let Some(content) = content else {
        return Ok(None);
    };

    let decoded = match content {
        Value::String(content) => serde_json::from_value(Value::String(content.clone()))
            .or_else(|_| serde_json::from_str(&content)),
        content => serde_json::from_value(content),
    };

    decoded.map(Some).map_err(D::Error::custom)
}

impl Message {
    fn new(role: Role, content: String, tool_call_id: Option<String>) -> Self {
        Self {
            id: new_uuid(),
            role,
            content: Some(content),
            thinking: None,
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

    pub fn with_image<T: Into<String>>(mut self, base64: T) -> Self {
        match self.images {
            Some(_) => todo!(),
            None => self.images = Some(vec![base64.into()]),
        }
        self
    }

    pub fn with_images<T: Into<String>>(mut self, base64: Vec<T>) -> Self {
        for image in base64 {
            self = self.with_image(image)
        }
        self
    }
}

fn new_uuid() -> String {
    Uuid::new_v4().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Deserialize, PartialEq)]
    struct Entity {
        name: String,
    }

    #[test]
    fn serde_decodes_json_string_content_into_the_message_type() {
        let message: Message<Entity> =
            serde_json::from_str(r#"{"role":"assistant","content":"{\"name\":\"Ada\"}"}"#).unwrap();

        assert_eq!(message.content, Some(Entity { name: "Ada".into() }));
    }

    #[test]
    fn serde_keeps_regular_content_as_text() {
        let message: Message =
            serde_json::from_str(r#"{"role":"assistant","content":"hello"}"#).unwrap();

        assert_eq!(message.content.as_deref(), Some("hello"));
    }
}
