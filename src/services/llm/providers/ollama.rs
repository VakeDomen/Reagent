use async_stream::try_stream;
use futures::{Stream, StreamExt};
use reqwest::Client;
use serde::de::DeserializeOwned;
use std::{fmt, pin::Pin};
use tracing::{debug, error, info_span, instrument, trace, Instrument};

use crate::services::llm::models::chat::ChatStreamChunk;
use crate::services::llm::models::{
    chat::{ChatRequest, ChatResponse},
    embedding::{EmbeddingsRequest, EmbeddingsResponse},
    errors::ModelClientError,
};

#[derive(Debug, Clone)]
pub struct OllamaClient {
    pub client: Client,
    pub base_url: String,
}

impl OllamaClient {
    pub fn new(cfg: crate::services::llm::client::ClientConfig) -> Result<Self, ModelClientError> {
        let base_url = cfg.base_url.unwrap_or("http://localhost:11434".into());
        Ok(Self {
            client: Client::new(),
            base_url,
        })
    }

    #[instrument(name = "ollama.post", skip_all, fields(endpoint))]
    async fn post<T, R>(&self, endpoint: &str, request_body: &T) -> Result<R, ModelClientError>
    where
        T: serde::Serialize + fmt::Debug,
        R: DeserializeOwned + fmt::Debug,
    {
        let url = format!("{}{}", self.base_url, endpoint);
        let span = info_span!("http.request", %url);
        async {
            println!(
                "{:#?}",
                serde_json::to_string_pretty(&request_body).unwrap()
            );

            let response = self
                .client
                .post(&url)
                .json(request_body)
                .send()
                .await
                .map_err(|e| ModelClientError::Api(e.to_string()))?;

            let status = response.status();
            debug!(%status, "received response");

            if !status.is_success() {
                let error_text = response
                    .text()
                    .await
                    .unwrap_or_else(|_| "Failed to read error body".into());
                error!(%status, body = %error_text, "request failed");
                return Err(ModelClientError::Api(format!(
                    "Request failed: {status} - {error_text}"
                )));
            }

            let response_text = response
                .text()
                .await
                .map_err(|e| ModelClientError::Api(format!("Failed to read response text: {e}")))?;

            match serde_json::from_str::<R>(&response_text) {
                Ok(parsed) => {
                    trace!(?parsed, "deserialized response");
                    Ok(parsed)
                }
                Err(e) => {
                    error!(%e, raw = %response_text, "deserialization error");
                    Err(ModelClientError::Serialization(format!(
                        "Error decoding response body: {e}. Raw JSON was: '{response_text}'"
                    )))
                }
            }
        }
        .instrument(span)
        .await
    }

    #[instrument(name = "ollama.post_stream", skip_all, fields(endpoint))]
    async fn post_stream<T, R>(
        &self,
        endpoint: &str,
        body: &T,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<R, ModelClientError>> + Send + 'static>>,
        ModelClientError,
    >
    where
        T: serde::Serialize + fmt::Debug,
        R: serde::de::DeserializeOwned + fmt::Debug + Send + 'static,
    {
        let url = format!("{}{}", self.base_url, endpoint);
        let resp = self
            .client
            .post(&url)
            .json(body)
            .send()
            .await
            .map_err(|e| ModelClientError::Api(e.to_string()))?;

        if !resp.status().is_success() {
            return Err(ModelClientError::Api(format!("Request failed: {resp:#?}")));
        }

        let byte_stream = resp.bytes_stream();
        let s = try_stream! {
            let mut buf = Vec::<u8>::new();
            futures::pin_mut!(byte_stream);
            while let Some(chunk) = byte_stream.next().await {
                let chunk = chunk.map_err(|e| ModelClientError::Request(e.to_string()))?;
                buf.extend_from_slice(&chunk);
                while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
                    let line: Vec<u8> = buf.drain(..=pos).collect();
                    let line = &line[..line.len() - 1];
                    if line.is_empty() { continue; }
                    let parsed: R = serde_json::from_slice(line)
                        .map_err(|e| ModelClientError::Serialization(e.to_string()))?;
                    yield parsed;
                }
            }
        };
        Ok(Box::pin(s))
    }

    pub async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, ModelClientError> {
        self.post("/api/chat", &request).await
    }

    pub async fn chat_stream(
        &self,
        req: ChatRequest,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<ChatStreamChunk, ModelClientError>> + Send + 'static>>,
        ModelClientError,
    > {
        self.post_stream("/api/chat", &req).await
    }

    pub async fn embeddings(
        &self,
        request: EmbeddingsRequest,
    ) -> Result<EmbeddingsResponse, ModelClientError> {
        self.post("/api/embeddings", &request).await
    }
}
