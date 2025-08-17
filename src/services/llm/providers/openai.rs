use std::pin::Pin;
use futures::Stream;

use crate::services::llm::client::ClientConfig;
use crate::services::llm::models::{
    chat::{ChatRequest, ChatResponse, ChatStreamChunk},
    embedding::{EmbeddingsRequest, EmbeddingsResponse},
    errors::ModelClientError,
};

#[derive(Debug, Clone)]
pub struct OpenAiClient {
    _cfg: ClientConfig,
}

impl OpenAiClient {
    pub fn new(_cfg: ClientConfig) -> Result<Self, ModelClientError> {
        Err(ModelClientError::Unsupported("OpenAI chat not implemented yet".into()))
    }

    pub async fn chat(&self, _req: ChatRequest) -> Result<ChatResponse, ModelClientError> {
        Err(ModelClientError::Unsupported("OpenAI chat not implemented yet".into()))
    }

    pub async fn chat_stream(
        &self,
        _req: ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatStreamChunk, ModelClientError>> + Send + 'static>>, ModelClientError> {
        Err(ModelClientError::Unsupported("OpenAI streaming not implemented yet".into()))
    }

    pub async fn embeddings(&self, _req: EmbeddingsRequest) -> Result<EmbeddingsResponse, ModelClientError> {
        Err(ModelClientError::Unsupported("OpenAI embeddings not implemented yet".into()))
    }
}
