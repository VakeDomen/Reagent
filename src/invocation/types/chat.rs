use rmcp::schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::model::execution;
use crate::services::llm::models::chat::ChatRequest;
use crate::{
    services::llm::{BaseRequest, ClientBuilder, ResponseFormatConfig},
    ChatResponse, InferenceOptions, Invocation, InvocationError, Message,
    NotificationOutputChannel, SchemaSpec, Tool,
};
use crate::{Standard, Structured};

/// Provider-neutral input for one chat/completion call.
#[derive(Debug, Clone)]
pub struct Chat {
    pub(crate) messages: Vec<Message>,
    pub(crate) tools: Option<Vec<Tool>>,
    pub(crate) options: InferenceOptions,
    pub(crate) response_format: ResponseFormatConfig,
    pub(crate) provider_format: Option<Value>,
    pub(crate) stream: Option<bool>,
    pub(crate) keep_alive: Option<String>,
}

impl Default for Chat {
    fn default() -> Self {
        Self {
            messages: Vec::new(),
            tools: None,
            options: InferenceOptions::default(),
            response_format: ResponseFormatConfig::default(),
            provider_format: None,
            // Providers do not agree on the default when this field is omitted.
            // Keep invocation execution and the request sent on the wire aligned.
            stream: Some(false),
            keep_alive: None,
        }
    }
}

pub type ChatInvocation = Invocation<Chat>;

impl Default for Invocation<Chat> {
    fn default() -> Self {
        Self::chat()
    }
}

impl Invocation<Chat, Standard> {
    pub fn chat() -> Self {
        Self::new(Chat::default())
    }
}

impl<O> Invocation<Chat, O> {
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

    pub(crate) fn set_response_format(mut self, schema: SchemaSpec) -> Self {
        self.request.response_format.set_spec(schema);
        self
    }

    pub(crate) fn set_response_format_str(mut self, schema: &str) -> Self {
        self.request.response_format.set_raw(schema);
        self
    }

    pub(crate) fn set_response_format_value(mut self, schema: Value) -> Self {
        self.request.response_format.set_value(schema);
        self
    }

    pub(crate) fn set_response_format_config(mut self, config: ResponseFormatConfig) -> Self {
        self.request.response_format = config;
        self
    }

    pub fn response_format_from<T: JsonSchema>(mut self) -> Invocation<Chat, Structured<T>> {
        self.request.response_format.set_type::<T>();
        self.with_output()
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

    async fn invoke_raw(self) -> Result<ChatResponse, InvocationError> {
        let model = self.model.ok_or(InvocationError::ModelNotDefined)?;
        let client = self.client_config.build()?;
        let schema = self
            .request
            .response_format
            .resolve()
            .map_err(InvocationError::InvalidJsonSchema)?;
        let format = schema
            .map(|schema| client.structured_output_format(&schema))
            .transpose()?;
        let request = ChatRequest {
            base: BaseRequest {
                model: model.clone(),
                format: self.request.provider_format.or(format),
                options: self.request.options.into_option(),
                stream: self.request.stream,
                keep_alive: self.request.keep_alive,
            },
            messages: self.request.messages,
            tools: self.request.tools,
        };
        let notifications =
            NotificationOutputChannel::new(self.notification_channel, self.name.unwrap_or(model));
        if request.base.stream == Some(true) {
            execution::invoke_streaming(request, &client, notifications).await
        } else {
            execution::invoke_nonstreaming(request, &client, notifications).await
        }
    }
}

impl Invocation<Chat, Standard> {
    /// Set a provider-neutral structured-output schema and decode its JSON response as a value.
    pub fn response_format(self, schema: SchemaSpec) -> Invocation<Chat, Structured<Value>> {
        self.set_response_format(schema).with_output()
    }

    /// Set a JSON schema and decode its response as a JSON value.
    pub fn response_format_str(self, schema: &str) -> Invocation<Chat, Structured<Value>> {
        self.set_response_format_str(schema).with_output()
    }

    /// Set a JSON schema and decode its response as a JSON value.
    pub fn response_format_value(self, schema: Value) -> Invocation<Chat, Structured<Value>> {
        self.set_response_format_value(schema).with_output()
    }
}

impl<T> Invocation<Chat, Structured<T>> {
    /// Replace the provider-neutral schema while retaining this invocation's output type.
    pub fn response_format(self, schema: SchemaSpec) -> Self {
        self.set_response_format(schema)
    }

    /// Replace the JSON schema while retaining this invocation's output type.
    pub fn response_format_str(self, schema: &str) -> Self {
        self.set_response_format_str(schema)
    }

    /// Replace the JSON schema while retaining this invocation's output type.
    pub fn response_format_value(self, schema: Value) -> Self {
        self.set_response_format_value(schema)
    }
}

impl Invocation<Chat, Standard> {
    /// Execute this fully configured invocation once.
    pub async fn invoke(self) -> Result<ChatResponse, InvocationError> {
        self.invoke_raw().await
    }
}

impl<T: DeserializeOwned> Invocation<Chat, Structured<T>> {
    /// Execute this invocation and parse the assistant message content.
    pub async fn invoke(self) -> Result<ChatResponse<T>, InvocationError> {
        let response = self.invoke_raw().await?;
        let message_id = response.message.id.clone();
        let encoded = serde_json::to_value(response)
            .map_err(|error| InvocationError::InvalidStructuredOutput(error.to_string()))?;
        let mut response = serde_json::from_value::<ChatResponse<T>>(encoded)
            .map_err(|error| InvocationError::InvalidStructuredOutput(error.to_string()))?;
        response.message.id = message_id;
        Ok(response)
    }
}
