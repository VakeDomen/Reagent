use rmcp::schemars::JsonSchema;
use serde_json::Value;

use crate::{
    services::llm::ResponseFormatConfig, InferenceOptions, Invocation, Message, Notification,
    SchemaSpec, Tool,
};

/// Provider-neutral input for one chat/completion call.
#[derive(Debug, Clone, Default)]
pub struct Chat {
    pub(crate) messages: Vec<Message>,
    pub(crate) tools: Option<Vec<Tool>>,
    pub(crate) options: InferenceOptions,
    pub(crate) response_format: ResponseFormatConfig,
    pub(crate) provider_format: Option<Value>,
    pub(crate) stream: Option<bool>,
    pub(crate) keep_alive: Option<String>,
}

pub type ChatInvocation = Invocation<Chat>;

impl Default for Invocation<Chat> {
    fn default() -> Self {
        Self::chat()
    }
}

impl Invocation<Chat> {
    pub fn chat() -> Self {
        Self::new(Chat::default())
    }

    pub fn keep_alive(mut self, value: impl Into<String>) -> Self {
        self.request.keep_alive = Some(value.into());
        self
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
