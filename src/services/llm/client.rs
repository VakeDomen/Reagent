use std::{pin::Pin, sync::Arc};

use futures::Stream;

use crate::{
    services::llm::models::{
        chat::{ChatRequest, ChatResponse, ChatStreamChunk},
        embedding::{EmbeddingsRequest, EmbeddingsResponse},
        errors::InferenceClientError,
    },
    ClientConfig,
};

use super::providers::{
    anthropic::AnthropicClient, mistral::MistralClient, ollama::OllamaClient, openai::OpenAiClient,
    openrouter::OpenRouterClient,
};

#[derive(Debug, Clone, Default)]
pub enum Provider {
    #[default]
    Ollama,
    OpenAi,
    Mistral,
    Anthropic,
    OpenRouter,
}

#[derive(Debug, Clone)]
enum ClientInner {
    Ollama(OllamaClient),
    OpenAi(OpenAiClient),
    Mistral(MistralClient),
    Anthropic(AnthropicClient),
    OpenRouter(OpenRouterClient),
}

#[derive(Clone, Debug)]
pub struct InferenceClient {
    config: ClientConfig,
    inner: Arc<ClientInner>,
}

impl InferenceClient {
    pub fn get_config(&self) -> &ClientConfig {
        &self.config
    }

    pub async fn chat(&self, req: ChatRequest) -> Result<ChatResponse, InferenceClientError> {
        match &*self.inner {
            ClientInner::Ollama(c) => c.chat(req).await,
            ClientInner::OpenAi(c) => c.chat(req).await,
            ClientInner::Mistral(c) => c.chat(req).await,
            ClientInner::Anthropic(c) => c.chat(req).await,
            ClientInner::OpenRouter(c) => c.chat(req).await,
        }
    }

    pub async fn chat_stream(
        &self,
        req: ChatRequest,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<ChatStreamChunk, InferenceClientError>> + Send + 'static>>,
        InferenceClientError,
    > {
        match &*self.inner {
            ClientInner::Ollama(c) => c.chat_stream(req).await,
            ClientInner::OpenAi(c) => c.chat_stream(req).await,
            ClientInner::Mistral(c) => c.chat_stream(req).await,
            ClientInner::Anthropic(c) => c.chat_stream(req).await,
            ClientInner::OpenRouter(c) => c.chat_stream(req).await,
        }
    }

    pub async fn embeddings(
        &self,
        req: EmbeddingsRequest,
    ) -> Result<EmbeddingsResponse, InferenceClientError> {
        match &*self.inner {
            ClientInner::Ollama(c) => c.embeddings(req).await,
            ClientInner::OpenAi(c) => c.embeddings(req).await,
            ClientInner::Mistral(c) => c.embeddings(req).await,
            ClientInner::Anthropic(c) => c.embeddings(req).await,
            ClientInner::OpenRouter(c) => c.embeddings(req).await,
        }
    }
}

impl TryFrom<ClientConfig> for InferenceClient {
    type Error = InferenceClientError;

    fn try_from(cfg: ClientConfig) -> Result<Self, Self::Error> {
        let config = cfg.clone();
        let Some(provider) = cfg.provider.clone() else {
            return Err(InferenceClientError::Config("Provider not defined".into()));
        };
        let inner = match provider {
            Provider::Ollama => ClientInner::Ollama(OllamaClient::new(cfg)?),
            Provider::OpenAi => ClientInner::OpenAi(OpenAiClient::new(cfg)?),
            Provider::Mistral => ClientInner::Mistral(MistralClient::new(cfg)?),
            Provider::Anthropic => ClientInner::Anthropic(AnthropicClient::new(cfg)?),
            Provider::OpenRouter => ClientInner::OpenRouter(OpenRouterClient::new(cfg)?),
        };
        Ok(Self {
            config,
            inner: Arc::new(inner),
        })
    }
}
