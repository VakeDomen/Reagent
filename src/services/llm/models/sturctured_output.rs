use rmcp::schemars::{schema_for, JsonSchema};
use serde_json::Value;

#[derive(Clone, Debug)]
pub struct SchemaSpec {
    pub schema: Value, // pure JSON Schema root
    pub name: Option<String>, // used by providers that want a name
    pub strict: Option<bool>, // opt-in, only applied where supported
}

impl SchemaSpec {
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
    pub fn strict(mut self, strict: bool) -> Self {
        self.strict = Some(strict);
        self
    }

    pub fn from_value(schema: serde_json::Value) -> Self {
        Self { schema, name: None, strict: None }
    }
    pub fn from_str(s: &str) -> Result<Self, serde_json::Error> {
        let v: serde_json::Value = serde_json::from_str(s.trim())?;
        Ok(Self::from_value(v))
    }
    pub fn from_type<T: JsonSchema>() -> Self {
        let schema = serde_json::to_value(schema_for!(T)).expect("schema serialize");
        Self::from_value(schema)
    }
}


pub fn to_ollama_format(spec: &SchemaSpec) -> serde_json::Value {
    spec.schema.clone()
}

pub fn to_openrouter_format(spec: &SchemaSpec) -> serde_json::Value {
    serde_json::json!({
        "type": "json_schema",
        "json_schema": {
            "name": spec.name.clone().unwrap_or_else(|| "schema".to_string()),
            "strict": spec.strict.unwrap_or(false),
            "schema": spec.schema
        }
    })
}