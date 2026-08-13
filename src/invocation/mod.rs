mod conversions;
mod error;
mod model;
mod types;

pub use error::*;
pub use model::*;
pub use types::*;

#[test]
fn chat_and_embedding_have_distinct_payloads() {
    use crate::Message;

    let chat = Invocation::chat().message(Message::user("hello"));
    let embedding = Invocation::embeddings(["one", "two"]);

    assert_eq!(chat.request.messages.len(), 1);
    assert_eq!(embedding.request.input, ["one", "two"]);
}

#[test]
fn tool_policy_is_resolved_before_invocation() {
    assert!(Invocation::chat().request.tools.is_none());
}

#[test]
fn chat_invocations_are_non_streaming_unless_explicitly_enabled() {
    assert_eq!(Invocation::chat().request.stream, Some(false));
    assert_eq!(Invocation::chat().stream(true).request.stream, Some(true));
}

#[test]
fn simple_values_convert_to_invocations() {
    let chat: ChatInvocation = "hello".into();
    let embedding: EmbeddingInvocation = ["one", "two"].into();

    assert_eq!(chat.request.messages.len(), 1);
    assert_eq!(embedding.request.input, ["one", "two"]);
}

#[test]
fn response_schema_selects_the_structured_invocation_output() {
    #[derive(serde::Deserialize, rmcp::schemars::JsonSchema)]
    struct Entity {
        name: String,
    }

    let _: Invocation<Chat, Structured<Entity>> =
        Invocation::chat().response_format_from::<Entity>();
}

#[test]
fn runtime_response_schemas_decode_to_json_values() {
    let schema = crate::SchemaSpec::from_value(serde_json::json!({"type": "object"}));

    let _: Invocation<Chat, Structured<serde_json::Value>> =
        Invocation::chat().response_format(schema);
    let _: Invocation<Chat, Structured<serde_json::Value>> =
        Invocation::chat().response_format_str(r#"{"type":"object"}"#);
    let _: Invocation<Chat, Structured<serde_json::Value>> =
        Invocation::chat().response_format_value(serde_json::json!({"type": "object"}));
}
