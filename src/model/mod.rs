use std::{collections::HashMap, marker::PhantomData};

use crate::{
    invocation::{Chat as ChatRequest, Embedding as EmbeddingRequest, Invocation, InvocationError},
    services::llm::{
        models::{
            chat::{ChatRequest as ChatRequestWire, ChatResponse},
            embedding::{EmbeddingsRequest, EmbeddingsResponse},
        },
        BaseRequest, ClientBuilder, ClientConfig, InferenceClient, InferenceOptions,
    },
    NotificationOutputChannel, Provider,
};

mod config;
mod execution;

pub use config::*;

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
pub type LlmModelBuilder = ModelBuilder<Llm>;
pub type EmbeddingModelBuilder = ModelBuilder<Embedding>;

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
                keep_alive: invocation.keep_alive.or_else(|| self.keep_alive.clone()),
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
            keep_alive: invocation.keep_alive.or_else(|| self.keep_alive.clone()),
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

    pub fn export_config(&self) -> ModelConfig {
        ModelConfig::from_parts(self.id.clone(), self.options.clone())
    }
}

/// Builds a reusable [`Model`]. Invocation inputs intentionally do not belong
/// here; they are supplied to [`Model::invoke`] for each call.
#[derive(Debug, Clone)]
pub struct ModelBuilder<M = Llm> {
    id: Option<String>,
    client_config: ClientConfig,
    options: InferenceOptions,
    stream: bool,
    keep_alive: Option<String>,
    kind: PhantomData<M>,
}

impl<M> Default for ModelBuilder<M> {
    fn default() -> Self {
        Self {
            id: None,
            client_config: ClientConfig::default(),
            options: InferenceOptions::default(),
            stream: false,
            keep_alive: None,
            kind: PhantomData,
        }
    }
}

impl<M> ModelBuilder<M> {
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

    pub fn build(self) -> Result<Model<M>, InvocationError> {
        let id = self.id.ok_or(InvocationError::ModelNotDefined)?;
        let client = self.client_config.build()?;

        Ok(Model {
            id,
            client,
            options: self.options,
            stream: self.stream,
            keep_alive: self.keep_alive,
            kind: PhantomData,
        })
    }
}

impl ModelBuilder<Llm> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_requires_an_identifier() {
        assert!(matches!(
            ModelBuilder::<Llm>::default().build(),
            Err(InvocationError::ModelNotDefined)
        ));
    }

    #[test]
    fn model_keeps_reusable_defaults_but_no_call_input() {
        let model: Model<Llm> = Model::llm("test-model")
            .temperature(0.2)
            .stream(true)
            .build()
            .unwrap();

        assert_eq!(model.id(), "test-model");
        assert_eq!(model.options().temperature, Some(0.2));
        assert!(model.stream_by_default());
    }

    #[test]
    fn model_kind_is_selected_at_build_time() {
        let _: Model<Llm> = Model::llm("chat-model").build().unwrap();
        let _: Model<Embedding> = Model::embedding("embedding-model").build().unwrap();
    }
}
