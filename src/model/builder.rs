use std::{collections::HashMap, marker::PhantomData};

use crate::{
    services::llm::{ClientBuilder, ResponseFormatConfig},
    ClientConfig, Embedding, InferenceOptions, InvocationError, Llm, Message, Model, Notification,
    Provider, SchemaSpec, Standard, Structured, Tool,
};
use serde_json::Value;
use tokio::sync::mpsc::Sender;

pub type LlmModelBuilder = ModelBuilder<Llm, Standard>;
pub type EmbeddingModelBuilder = ModelBuilder<Embedding>;

/// Builds a reusable [`Model`]. Invocation inputs intentionally do not belong
/// here; they are supplied to [`Model::invoke`] for each call.
#[derive(Debug, Clone)]
pub struct ModelBuilder<M = Llm, O = Standard> {
    id: Option<String>,
    client_config: ClientConfig,
    options: InferenceOptions,
    stream: bool,
    keep_alive: Option<String>,
    history: Vec<Message>,
    tools: Option<Vec<Tool>>,
    response_format: ResponseFormatConfig,
    provider_format: Option<Value>,
    name: Option<String>,
    notification_channel: Option<Sender<Notification>>,
    kind: PhantomData<(M, O)>,
}

impl<M, O> Default for ModelBuilder<M, O> {
    fn default() -> Self {
        Self {
            id: None,
            client_config: ClientConfig::default(),
            options: InferenceOptions::default(),
            stream: false,
            keep_alive: None,
            history: Vec::new(),
            tools: None,
            response_format: ResponseFormatConfig::default(),
            provider_format: None,
            name: None,
            notification_channel: None,
            kind: PhantomData,
        }
    }
}

impl<M, O> ModelBuilder<M, O> {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: Some(id.into()),
            ..Self::default()
        }
    }

    pub fn model(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    pub fn client_config(mut self, config: ClientConfig) -> Self {
        self.client_config = config;
        self
    }

    pub fn provider(mut self, provider: Provider) -> Self {
        self.client_config = self.client_config.provider(Some(provider));
        self
    }

    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.client_config = self.client_config.base_url(Some(base_url));
        self
    }

    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.client_config = self.client_config.api_key(Some(api_key));
        self
    }

    pub fn organization(mut self, organization: impl Into<String>) -> Self {
        self.client_config = self.client_config.organization(Some(organization));
        self
    }

    pub fn extra_headers(mut self, headers: HashMap<String, String>) -> Self {
        self.client_config = self.client_config.extra_headers(Some(headers));
        self
    }

    pub fn keep_alive(mut self, keep_alive: impl Into<String>) -> Self {
        self.keep_alive = Some(keep_alive.into());
        self
    }

    pub fn build(self) -> Result<Model<M, O>, InvocationError> {
        let id = self.id.ok_or(InvocationError::ModelNotDefined)?;
        self.client_config.clone().build()?;

        Ok(Model::new(
            id,
            self.client_config,
            self.options,
            self.stream,
            self.keep_alive,
            self.history,
            self.tools,
            self.response_format,
            self.provider_format,
            self.name,
            self.notification_channel,
            self.kind,
        ))
    }
}

impl<O> ModelBuilder<Llm, O> {
    pub fn set_history(mut self, history: impl Into<Vec<Message>>) -> Self {
        self.history = history.into();
        self
    }

    pub fn tools(mut self, tools: Vec<Tool>) -> Self {
        self.tools = Some(tools);
        self
    }

    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn notification_channel(mut self, channel: Option<Sender<Notification>>) -> Self {
        self.notification_channel = channel;
        self
    }

    /// Escape hatch for an already provider-formatted response format.
    pub fn provider_format(mut self, format: Value) -> Self {
        self.provider_format = Some(format);
        self
    }

    pub fn schema_name(mut self, name: impl Into<String>) -> Self {
        self.response_format.set_name(name);
        self
    }

    pub fn schema_strict(mut self, strict: bool) -> Self {
        self.response_format.set_strict(strict);
        self
    }

    pub fn options(mut self, options: InferenceOptions) -> Self {
        self.options = options;
        self
    }

    pub fn stream(mut self, stream: bool) -> Self {
        self.stream = stream;
        self
    }

    pub fn temperature(mut self, value: f32) -> Self {
        self.options.temperature = Some(value);
        self
    }

    pub fn top_p(mut self, value: f32) -> Self {
        self.options.top_p = Some(value);
        self
    }

    pub fn presence_penalty(mut self, value: f32) -> Self {
        self.options.presence_penalty = Some(value);
        self
    }

    pub fn frequency_penalty(mut self, value: f32) -> Self {
        self.options.frequency_penalty = Some(value);
        self
    }

    pub fn num_ctx(mut self, value: u32) -> Self {
        self.options.num_ctx = Some(value);
        self
    }

    pub fn repeat_last_n(mut self, value: i32) -> Self {
        self.options.repeat_last_n = Some(value);
        self
    }

    pub fn repeat_penalty(mut self, value: f32) -> Self {
        self.options.repeat_penalty = Some(value);
        self
    }

    pub fn seed(mut self, value: i32) -> Self {
        self.options.seed = Some(value);
        self
    }

    pub fn stop(mut self, value: impl Into<String>) -> Self {
        self.options.stop = Some(value.into());
        self
    }

    pub fn num_predict(mut self, value: i32) -> Self {
        self.options.num_predict = Some(value);
        self
    }

    pub fn max_tokens(mut self, value: i32) -> Self {
        self.options.max_tokens = Some(value);
        self
    }

    pub fn top_k(mut self, value: u32) -> Self {
        self.options.top_k = Some(value);
        self
    }

    pub fn min_p(mut self, value: f32) -> Self {
        self.options.min_p = Some(value);
        self
    }
}

impl ModelBuilder<Llm, Standard> {
    pub fn structured_output<T: rmcp::schemars::JsonSchema>(
        self,
    ) -> ModelBuilder<Llm, Structured<T>> {
        ModelBuilder {
            id: self.id,
            client_config: self.client_config,
            options: self.options,
            stream: self.stream,
            keep_alive: self.keep_alive,
            history: self.history,
            tools: self.tools,
            response_format: {
                let mut format = self.response_format;
                format.set_type::<T>();
                format
            },
            provider_format: self.provider_format,
            name: self.name,
            notification_channel: self.notification_channel,
            kind: PhantomData,
        }
    }

    pub fn response_format(mut self, schema: SchemaSpec) -> ModelBuilder<Llm, Structured<Value>> {
        self.response_format.set_spec(schema);
        self.with_structured_value()
    }

    pub fn response_format_str(mut self, schema: &str) -> ModelBuilder<Llm, Structured<Value>> {
        self.response_format.set_raw(schema);
        self.with_structured_value()
    }

    pub fn response_format_value(mut self, schema: Value) -> ModelBuilder<Llm, Structured<Value>> {
        self.response_format.set_value(schema);
        self.with_structured_value()
    }

    pub fn response_format_from<T: rmcp::schemars::JsonSchema>(
        self,
    ) -> ModelBuilder<Llm, Structured<T>> {
        self.structured_output()
    }

    fn with_structured_value(self) -> ModelBuilder<Llm, Structured<Value>> {
        ModelBuilder {
            id: self.id,
            client_config: self.client_config,
            options: self.options,
            stream: self.stream,
            keep_alive: self.keep_alive,
            history: self.history,
            tools: self.tools,
            response_format: self.response_format,
            provider_format: self.provider_format,
            name: self.name,
            notification_channel: self.notification_channel,
            kind: PhantomData,
        }
    }
}

impl<O> ModelBuilder<Llm, Structured<O>> {
    pub fn response_format(mut self, schema: SchemaSpec) -> Self {
        self.response_format.set_spec(schema);
        self
    }

    pub fn response_format_str(mut self, schema: &str) -> Self {
        self.response_format.set_raw(schema);
        self
    }

    pub fn response_format_value(mut self, schema: Value) -> Self {
        self.response_format.set_value(schema);
        self
    }
}
