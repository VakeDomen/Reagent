use std::{collections::HashMap, marker::PhantomData};

use crate::{
    services::llm::ClientBuilder, ClientConfig, Embedding, InferenceOptions, InvocationError, Llm,
    Message, Model, Provider, SchemaSpec, Standard, Structured,
};

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
    response_format: Option<SchemaSpec>,
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
            response_format: None,
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
            None,
            self.response_format,
            None,
            None,
            self.kind,
        ))
    }
}

impl ModelBuilder<Llm, Standard> {
    pub fn set_history(mut self, history: impl Into<Vec<Message>>) -> Self {
        self.history = history.into();
        self
    }

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
            response_format: Some(SchemaSpec::from_type::<T>()),
            kind: PhantomData,
        }
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
