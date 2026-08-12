use std::marker::PhantomData;

use crate::{
    invocation::{Chat as ChatRequest, Embedding as EmbeddingRequest, Invocation, InvocationError},
    services::llm::{
        models::{
            chat::{ChatRequest as ChatRequestWire, ChatResponse},
            embedding::{EmbeddingsRequest, EmbeddingsResponse},
        },
        BaseRequest, ClientConfig, InferenceClient, InferenceOptions,
    },
    ModelBuilder, NotificationOutputChannel,
};

use super::execution;

/// A reusable, sessionless inference model.
///
/// `Model` owns immutable endpoint and inference defaults. Calling it never
/// records messages or changes subsequent calls, so it can be cloned and shared
/// safely by tasks and agents.
#[derive(Debug, Clone, Default)]
pub struct Llm;

#[derive(Debug, Clone, Default)]
pub struct Embedding;

pub type LlmModel = Model<Llm>;
pub type EmbeddingModel = Model<Embedding>;

#[derive(Clone, Debug)]
pub struct Model<M = Llm> {
    id: String,
    client: InferenceClient,
    options: InferenceOptions,
    stream: bool,
    keep_alive: Option<String>,
    kind: PhantomData<M>,
}

impl Model<Llm> {
    pub fn llm(id: impl Into<String>) -> ModelBuilder<Llm> {
        ModelBuilder::new(id)
    }

    /// Alias for [`Model::llm`]. A default `Model` is an LLM model.
    pub fn builder(id: impl Into<String>) -> ModelBuilder<Llm> {
        Self::llm(id)
    }

    /// Execute one chat invocation without retaining any state from it.
    pub async fn invoke(
        &self,
        invocation: impl Into<Invocation<ChatRequest>>,
    ) -> Result<ChatResponse, InvocationError> {
        let invocation = invocation.into();
        let schema = invocation
            .request
            .response_format
            .resolve()
            .map_err(InvocationError::InvalidJsonSchema)?;
        let format = schema
            .map(|schema| self.client.structured_output_format(&schema))
            .transpose()?;

        let request = ChatRequestWire {
            base: BaseRequest {
                model: self.id.clone(),
                format: invocation.request.provider_format.or(format),
                options: invocation
                    .request
                    .options
                    .merge_over(self.options.clone())
                    .into_option(),
                stream: Some(invocation.request.stream.unwrap_or(self.stream)),
                keep_alive: invocation
                    .request
                    .keep_alive
                    .or_else(|| self.keep_alive.clone()),
            },
            messages: invocation.request.messages,
            tools: invocation.request.tools,
        };

        let notifications = NotificationOutputChannel::new(
            invocation.notification_channel,
            invocation.name.unwrap_or_else(|| self.id.clone()),
        );

        if request.base.stream == Some(true) {
            execution::invoke_streaming(request, &self.client, notifications).await
        } else {
            execution::invoke_nonstreaming(request, &self.client, notifications).await
        }
    }
}

impl Model<Embedding> {
    pub fn embedding(id: impl Into<String>) -> ModelBuilder<Embedding> {
        ModelBuilder::new(id)
    }

    /// Execute one embedding invocation without retaining any state from it.
    pub async fn invoke(
        &self,
        invocation: impl Into<Invocation<EmbeddingRequest>>,
    ) -> Result<EmbeddingsResponse, InvocationError> {
        let invocation = invocation.into();
        if invocation.request.input.is_empty() {
            return Err(InvocationError::InputNotDefined);
        }

        let request = EmbeddingsRequest {
            model: self.id.clone(),
            input: invocation.request.input,
            options: None,
            keep_alive: invocation
                .request
                .keep_alive
                .or_else(|| self.keep_alive.clone()),
        };

        Ok(self.client.embeddings(request).await?)
    }
}

impl<M> Model<M> {
    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn client_config(&self) -> &ClientConfig {
        self.client.get_config()
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

    pub fn new(
        id: String,
        client: InferenceClient,
        options: InferenceOptions,
        stream: bool,
        keep_alive: Option<String>,
        kind: PhantomData<M>,
    ) -> Self {
        Self {
            id,
            client,
            options,
            stream,
            keep_alive,
            kind,
        }
    }
}
