mod chat;
mod conversions;
mod embedding;
mod error;
mod model;

pub use chat::*;
pub use embedding::*;
pub use error::*;
pub use model::*;

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
fn simple_values_convert_to_invocations() {
    let chat: ChatInvocation = "hello".into();
    let embedding: EmbeddingInvocation = ["one", "two"].into();

    assert_eq!(chat.request.messages.len(), 1);
    assert_eq!(embedding.request.input, ["one", "two"]);
}
