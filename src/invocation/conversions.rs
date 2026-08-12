use super::{Chat, Embedding, Invocation};
use crate::Message;

impl From<String> for Invocation<Embedding> {
    fn from(input: String) -> Self {
        Invocation::embedding(input)
    }
}
impl From<&str> for Invocation<Embedding> {
    fn from(input: &str) -> Self {
        Invocation::embedding(input)
    }
}
impl From<Vec<String>> for Invocation<Embedding> {
    fn from(inputs: Vec<String>) -> Self {
        Invocation::embeddings(inputs)
    }
}
impl From<Vec<&str>> for Invocation<Embedding> {
    fn from(inputs: Vec<&str>) -> Self {
        Invocation::embeddings(inputs)
    }
}
impl<const N: usize> From<[String; N]> for Invocation<Embedding> {
    fn from(inputs: [String; N]) -> Self {
        Invocation::embeddings(inputs)
    }
}
impl<const N: usize> From<[&str; N]> for Invocation<Embedding> {
    fn from(inputs: [&str; N]) -> Self {
        Invocation::embeddings(inputs)
    }
}
impl From<Message> for Invocation<Chat> {
    fn from(message: Message) -> Self {
        Invocation::chat().message(message)
    }
}
impl From<Vec<Message>> for Invocation<Chat> {
    fn from(messages: Vec<Message>) -> Self {
        Invocation::chat().messages(messages)
    }
}
impl From<String> for Invocation<Chat> {
    fn from(prompt: String) -> Self {
        Invocation::chat().message(Message::user(prompt))
    }
}
impl From<&str> for Invocation<Chat> {
    fn from(prompt: &str) -> Self {
        Invocation::chat().message(Message::user(prompt))
    }
}

impl From<String> for Message {
    fn from(prompt: String) -> Self {
        Message::user(prompt)
    }
}

impl From<&str> for Message {
    fn from(prompt: &str) -> Self {
        Message::user(prompt)
    }
}
