use rmcp::schemars::JsonSchema;
use serde_json::Value;
use tokio::sync::mpsc::Sender;

use crate::{
    services::llm::{InferenceOptions, ResponseFormatConfig, SchemaSpec},
    Message, Notification, Tool,
};

mod error;

pub use error::*;

/// Provider-neutral input for one chat/completion call.
#[derive(Debug, Clone, Default)]
pub struct Chat {
    pub(crate) messages: Vec<Message>,
    pub(crate) tools: Option<Vec<Tool>>,
    pub(crate) options: InferenceOptions,
    pub(crate) response_format: ResponseFormatConfig,
    pub(crate) provider_format: Option<Value>,
    pub(crate) stream: Option<bool>,
}

/// Provider-neutral input for one embedding call.
#[derive(Debug, Clone, Default)]
pub struct Embedding {
    pub(crate) input: Vec<String>,
}

/// One passive, provider-neutral request to a [`crate::Model`].
///
/// An invocation contains only data for a single call. It does not own a client,
/// model identifier, conversation history, or any state that can survive the call.
#[derive(Debug, Clone)]
pub struct Invocation<R = Chat> {
    pub(crate) request: R,
    pub(crate) keep_alive: Option<String>,
    pub(crate) name: Option<String>,
    pub(crate) notification_channel: Option<Sender<Notification>>,
}

pub type ChatInvocation = Invocation<Chat>;
pub type EmbeddingInvocation = Invocation<Embedding>;

impl<R> Invocation<R> {
    fn new(request: R) -> Self {
        Self {
            request,
            keep_alive: None,
            name: None,
            notification_channel: None,
        }
    }

    /// Override the model's keep-alive setting for this call.
    pub fn keep_alive(mut self, value: impl Into<String>) -> Self {
        self.keep_alive = Some(value.into());
        self
    }

    /// Set the name used for notifications and tracing for this call.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Send events produced by this call to the supplied channel.
    pub fn notification_channel(mut self, channel: Option<Sender<Notification>>) -> Self {
        self.notification_channel = channel;
        self
    }
}

impl Default for Invocation<Chat> {
    fn default() -> Self {
        Self::chat()
    }
}

impl Invocation<Chat> {
    pub fn chat() -> Self {
        Self::new(Chat::default())
    }

    pub fn messages(mut self, messages: Vec<Message>) -> Self {
        self.request.messages = messages;
        self
    }

    pub fn message(mut self, message: Message) -> Self {
        self.request.messages.push(message);
        self
    }

    pub fn tools(mut self, tools: Vec<Tool>) -> Self {
        self.request.tools = Some(tools);
        self
    }

    pub fn stream(mut self, stream: bool) -> Self {
        self.request.stream = Some(stream);
        self
    }

    pub fn options(mut self, options: InferenceOptions) -> Self {
        self.request.options = options;
        self
    }

    pub fn num_ctx(mut self, value: u32) -> Self {
        self.request.options.num_ctx = Some(value);
        self
    }

    pub fn repeat_last_n(mut self, value: i32) -> Self {
        self.request.options.repeat_last_n = Some(value);
        self
    }

    pub fn repeat_penalty(mut self, value: f32) -> Self {
        self.request.options.repeat_penalty = Some(value);
        self
    }

    pub fn temperature(mut self, value: f32) -> Self {
        self.request.options.temperature = Some(value);
        self
    }

    pub fn seed(mut self, value: i32) -> Self {
        self.request.options.seed = Some(value);
        self
    }

    pub fn stop(mut self, value: impl Into<String>) -> Self {
        self.request.options.stop = Some(value.into());
        self
    }

    pub fn num_predict(mut self, value: i32) -> Self {
        self.request.options.num_predict = Some(value);
        self
    }

    pub fn max_tokens(mut self, value: i32) -> Self {
        self.request.options.max_tokens = Some(value);
        self
    }

    pub fn top_k(mut self, value: u32) -> Self {
        self.request.options.top_k = Some(value);
        self
    }

    pub fn top_p(mut self, value: f32) -> Self {
        self.request.options.top_p = Some(value);
        self
    }

    pub fn min_p(mut self, value: f32) -> Self {
        self.request.options.min_p = Some(value);
        self
    }

    pub fn presence_penalty(mut self, value: f32) -> Self {
        self.request.options.presence_penalty = Some(value);
        self
    }

    pub fn frequency_penalty(mut self, value: f32) -> Self {
        self.request.options.frequency_penalty = Some(value);
        self
    }

    /// Set a provider-neutral structured-output schema.
    pub fn response_format(mut self, schema: SchemaSpec) -> Self {
        self.request.response_format.set_spec(schema);
        self
    }

    pub fn response_format_str(mut self, schema: &str) -> Self {
        self.request.response_format.set_raw(schema);
        self
    }

    pub fn response_format_value(mut self, schema: Value) -> Self {
        self.request.response_format.set_value(schema);
        self
    }

    pub fn response_format_from<T: JsonSchema>(mut self) -> Self {
        self.request.response_format.set_type::<T>();
        self
    }

    pub fn schema_name(mut self, name: impl Into<String>) -> Self {
        self.request.response_format.set_name(name);
        self
    }

    pub fn schema_strict(mut self, strict: bool) -> Self {
        self.request.response_format.set_strict(strict);
        self
    }

    /// Escape hatch for an already provider-formatted response format.
    pub fn provider_format(mut self, format: Value) -> Self {
        self.request.provider_format = Some(format);
        self
    }
}

impl Invocation<Embedding> {
    pub fn embedding(input: impl Into<String>) -> Self {
        Self::new(Embedding {
            input: vec![input.into()],
        })
    }

    pub fn embeddings<T, I>(inputs: I) -> Self
    where
        T: Into<String>,
        I: IntoIterator<Item = T>,
    {
        Self::new(Embedding {
            input: inputs.into_iter().map(Into::into).collect(),
        })
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_and_embedding_have_distinct_payloads() {
        let chat = Invocation::chat().message(Message::user("hello"));
        let embedding = Invocation::embeddings(["one", "two"]);

        assert_eq!(chat.request.messages.len(), 1);
        assert_eq!(embedding.request.input, ["one", "two"]);
    }

    #[test]
    fn tool_policy_is_resolved_before_invocation() {
        let invocation = Invocation::chat();
        assert!(invocation.request.tools.is_none());
    }

    #[test]
    fn simple_values_convert_to_invocations() {
        let chat: ChatInvocation = "hello".into();
        let embedding: EmbeddingInvocation = ["one", "two"].into();

        assert_eq!(chat.request.messages.len(), 1);
        assert_eq!(embedding.request.input, ["one", "two"]);
    }
}
