use std::marker::PhantomData;

use serde::de::DeserializeOwned;

use super::execution;
use crate::{
    services::llm::{
        models::{chat::ChatRequest, embedding::EmbeddingsRequest},
        BaseRequest, ClientBuilder,
    },
    ChatResponse, ClientConfig, EmbeddingsResponse, InferenceOptions, InvocationError, Message,
    ModelBuilder, NotificationOutputChannel, SchemaSpec,
};

/// A reusable, sessionless inference model.
///
/// `Model` owns reusable endpoint, inference, and optional history defaults.
/// Calling it never records its prompt or response, though its configuration can
/// be changed explicitly through mutable setters.
#[derive(Debug, Clone, Default)]
pub struct Llm;

#[derive(Debug, Clone, Default)]
pub struct Embedding;

/// The default LLM output: the provider response is returned unchanged.
#[derive(Debug, Clone, Default)]
pub struct Standard;

/// A schema-backed LLM output parsed directly into `T`.
#[derive(Debug, Clone, Default)]
pub struct Structured<T>(PhantomData<T>);

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
    response_format: Option<SchemaSpec>,
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
        prompt: impl Into<Message>,
    ) -> Result<ChatResponse, InvocationError> {
        let client = self.client_config.clone().build()?;
        let messages = self
            .history
            .iter()
            .cloned()
            .chain(std::iter::once(prompt.into()))
            .collect();
        let request = ChatRequest {
            base: BaseRequest {
                model: self.id.clone(),
                format: None,
                options: self.options.clone().into_option(),
                stream: Some(self.stream),
                keep_alive: self.keep_alive.clone(),
            },
            messages,
            tools: None,
        };
        let notifications = NotificationOutputChannel::new(None, self.id.clone());
        if self.stream {
            execution::invoke_streaming(request, &client, notifications).await
        } else {
            execution::invoke_nonstreaming(request, &client, notifications).await
        }
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
        let input = input.into().request.input;
        if input.is_empty() {
            return Err(InvocationError::InputNotDefined);
        }
        let client = self.client_config.clone().build()?;
        Ok(client
            .embeddings(EmbeddingsRequest {
                model: self.id.clone(),
                input,
                options: None,
                keep_alive: self.keep_alive.clone(),
            })
            .await?)
    }
}

impl<T: DeserializeOwned> Model<Llm, Structured<T>> {
    pub async fn invoke(&self, prompt: impl Into<Message>) -> Result<T, InvocationError> {
        let client = self.client_config.clone().build()?;
        let format = self
            .response_format
            .as_ref()
            .map(|schema| client.structured_output_format(schema))
            .transpose()?;
        let messages = self
            .history
            .iter()
            .cloned()
            .chain(std::iter::once(prompt.into()))
            .collect();
        let request = ChatRequest {
            base: BaseRequest {
                model: self.id.clone(),
                format,
                options: self.options.clone().into_option(),
                stream: Some(self.stream),
                keep_alive: self.keep_alive.clone(),
            },
            messages,
            tools: None,
        };
        let notifications = NotificationOutputChannel::new(None, self.id.clone());
        let response = if self.stream {
            execution::invoke_streaming(request, &client, notifications).await?
        } else {
            execution::invoke_nonstreaming(request, &client, notifications).await?
        };
        let content = response.message.content.ok_or_else(|| {
            InvocationError::InvalidStructuredOutput("model did not return content".into())
        })?;
        serde_json::from_str(&content)
            .map_err(|error| InvocationError::InvalidStructuredOutput(error.to_string()))
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
        response_format: Option<SchemaSpec>,
        kind: PhantomData<(M, O)>,
    ) -> Self {
        Self {
            id,
            client_config,
            options,
            stream,
            keep_alive,
            history,
            response_format,
            kind,
        }
    }
}
