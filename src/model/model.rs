use std::marker::PhantomData;

use serde::de::DeserializeOwned;

use crate::invocation::{Standard, Structured};
use crate::services::llm::ResponseFormatConfig;
use crate::{
    ChatResponse, ClientConfig, EmbeddingsResponse, InferenceOptions, Invocation, InvocationError,
    Message, ModelBuilder, Notification, SchemaSpec, Tool,
};
use serde_json::Value;
use tokio::sync::mpsc::Sender;

pub trait IntoModelInput {
    fn into_messages(self) -> Vec<Message>;
}
impl IntoModelInput for Message {
    fn into_messages(self) -> Vec<Message> {
        vec![self]
    }
}
impl IntoModelInput for String {
    fn into_messages(self) -> Vec<Message> {
        vec![Message::user(self)]
    }
}
impl IntoModelInput for &str {
    fn into_messages(self) -> Vec<Message> {
        vec![Message::user(self)]
    }
}
impl IntoModelInput for Vec<Message> {
    fn into_messages(self) -> Vec<Message> {
        self
    }
}
impl IntoModelInput for () {
    fn into_messages(self) -> Vec<Message> {
        Vec::new()
    }
}

/// A reusable, sessionless inference model.
///
/// `Model` owns reusable endpoint, inference, and optional history defaults.
/// Calling it never records its prompt or response, though its configuration can
/// be changed explicitly through mutable setters.
#[derive(Debug, Clone, Default)]
pub struct Llm;

#[derive(Debug, Clone, Default)]
pub struct Embedding;

pub type LlmModel = Model<Llm, Standard>;
pub type EmbeddingModel = Model<Embedding>;

#[derive(Clone, Debug)]
pub struct Model<M = Llm, O = Standard> {
    id: String,
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

impl Model<Llm, Standard> {
    pub fn llm(id: impl Into<String>) -> ModelBuilder<Llm> {
        ModelBuilder::new(id)
    }

    /// Alias for [`Model::llm`]. A default `Model` is an LLM model.
    pub fn builder(id: impl Into<String>) -> ModelBuilder<Llm> {
        Self::llm(id)
    }

    /// Execute a prompt using configured history plus one ephemeral user message.
    pub async fn invoke(
        &self,
        input: impl IntoModelInput,
    ) -> Result<ChatResponse, InvocationError> {
        let messages = self
            .history
            .iter()
            .cloned()
            .chain(input.into_messages())
            .collect();
        let mut invocation = Invocation::chat()
            .model(self.id.clone())
            .client_config(self.client_config.clone())
            .messages(messages)
            .options(self.options.clone())
            .stream(self.stream)
            .notification_channel(self.notification_channel.clone());
        if let Some(tools) = &self.tools {
            invocation = invocation.tools(tools.clone());
        }
        invocation = invocation.set_response_format_config(self.response_format.clone());
        if let Some(format) = &self.provider_format {
            invocation = invocation.provider_format(format.clone());
        }
        if let Some(keep_alive) = &self.keep_alive {
            invocation = invocation.keep_alive(keep_alive.clone());
        }
        if let Some(name) = &self.name {
            invocation = invocation.name(name.clone());
        }
        invocation.invoke().await
    }
}

impl Model<Embedding> {
    pub fn embedding(id: impl Into<String>) -> ModelBuilder<Embedding> {
        ModelBuilder::new(id)
    }

    /// Execute embedding input without retaining it.
    pub async fn invoke(
        &self,
        input: impl Into<crate::EmbeddingInvocation>,
    ) -> Result<EmbeddingsResponse, InvocationError> {
        let mut invocation = input
            .into()
            .model(self.id.clone())
            .client_config(self.client_config.clone());
        if let Some(keep_alive) = &self.keep_alive {
            invocation = invocation.keep_alive(keep_alive.clone());
        }
        invocation.invoke().await
    }
}

impl<T: DeserializeOwned> Model<Llm, Structured<T>> {
    pub async fn invoke(
        &self,
        input: impl IntoModelInput,
    ) -> Result<ChatResponse<T>, InvocationError> {
        let messages = self
            .history
            .iter()
            .cloned()
            .chain(input.into_messages())
            .collect();
        let mut invocation = Invocation::chat()
            .model(self.id.clone())
            .client_config(self.client_config.clone())
            .messages(messages)
            .options(self.options.clone())
            .stream(self.stream)
            .notification_channel(self.notification_channel.clone());
        if let Some(tools) = &self.tools {
            invocation = invocation.tools(tools.clone());
        }
        invocation = invocation.set_response_format_config(self.response_format.clone());
        if let Some(format) = &self.provider_format {
            invocation = invocation.provider_format(format.clone());
        }
        if let Some(keep_alive) = &self.keep_alive {
            invocation = invocation.keep_alive(keep_alive.clone());
        }
        if let Some(name) = &self.name {
            invocation = invocation.name(name.clone());
        }
        invocation.structured_output::<T>().invoke().await
    }
}

impl<O> Model<Llm, O> {
    pub fn history(&self) -> &[Message] {
        &self.history
    }
    pub fn set_history(&mut self, history: impl Into<Vec<Message>>) -> &mut Self {
        self.history = history.into();
        self
    }
    pub fn clear_history(&mut self) -> &mut Self {
        self.history.clear();
        self
    }
    pub fn set_tools(&mut self, tools: Option<Vec<Tool>>) -> &mut Self {
        self.tools = tools;
        self
    }
    pub fn set_response_format(&mut self, response_format: Option<SchemaSpec>) -> &mut Self {
        self.response_format = ResponseFormatConfig::default();
        if let Some(response_format) = response_format {
            self.response_format.set_spec(response_format);
        }
        self
    }
    pub fn set_name(&mut self, name: Option<String>) -> &mut Self {
        self.name = name;
        self
    }
    pub fn set_notification_channel(&mut self, channel: Option<Sender<Notification>>) -> &mut Self {
        self.notification_channel = channel;
        self
    }
    pub fn set_temperature(&mut self, value: f32) -> &mut Self {
        self.options.temperature = Some(value);
        self
    }
    pub fn set_stream(&mut self, stream: bool) -> &mut Self {
        self.stream = stream;
        self
    }
    pub fn set_keep_alive(&mut self, value: impl Into<String>) -> &mut Self {
        self.keep_alive = Some(value.into());
        self
    }
}

impl<M, O> Model<M, O> {
    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn client_config(&self) -> &ClientConfig {
        &self.client_config
    }

    pub fn options(&self) -> &InferenceOptions {
        &self.options
    }

    pub fn stream_by_default(&self) -> bool {
        self.stream
    }

    pub fn keep_alive(&self) -> Option<&str> {
        self.keep_alive.as_deref()
    }

    pub fn export_config(&self) -> InferenceOptions {
        self.options.clone()
    }

    pub(crate) fn new(
        id: String,
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
    ) -> Self {
        Self {
            id,
            client_config,
            options,
            stream,
            keep_alive,
            history,
            tools,
            response_format,
            provider_format,
            name,
            notification_channel,
            kind,
        }
    }
}
