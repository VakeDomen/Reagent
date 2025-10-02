use futures::Stream;
use std::pin::Pin;

use crate::{
    services::llm::models::{
        chat::{ChatRequest, ChatResponse, ChatStreamChunk},
        embedding::{EmbeddingsRequest, EmbeddingsResponse},
        errors::InferenceClientError,
    },
    ClientConfig,
};

#[derive(Debug, Clone)]
pub struct MistralClient {
    _cfg: ClientConfig,
}

impl MistralClient {
    pub fn new(_cfg: ClientConfig) -> Result<Self, InferenceClientError> {
        Err(InferenceClientError::Unsupported(
            "Mistral chat not implemented yet".into(),
        ))
    }

    pub async fn chat(&self, _req: ChatRequest) -> Result<ChatResponse, InferenceClientError> {
        Err(InferenceClientError::Unsupported(
            "Mistral chat not implemented yet".into(),
        ))
    }

    pub async fn chat_stream(
        &self,
        _req: ChatRequest,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<ChatStreamChunk, InferenceClientError>> + Send + 'static>>,
        InferenceClientError,
    > {
        Err(InferenceClientError::Unsupported(
            "Mistral streaming not implemented yet".into(),
        ))
    }

    pub async fn embeddings(
        &self,
        _req: EmbeddingsRequest,
    ) -> Result<EmbeddingsResponse, InferenceClientError> {
        Err(InferenceClientError::Unsupported(
            "Mistral embeddings not implemented yet".into(),
        ))
    }
}
