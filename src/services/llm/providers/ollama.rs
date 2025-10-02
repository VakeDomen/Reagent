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
    errors::InferenceClientError,
};
use crate::services::llm::StructuredOuputFormat;
use crate::ClientConfig;

#[derive(Debug, Clone)]
pub struct OllamaClient {
    pub client: Client,
    pub base_url: String,
}

impl OllamaClient {
    pub fn new(cfg: ClientConfig) -> Result<Self, InferenceClientError> {
        let base_url = cfg.base_url.unwrap_or("http://localhost:11434".into());
        Ok(Self {
            client: Client::new(),
            base_url,
        })
    }

    #[instrument(name = "ollama.post", skip_all, fields(endpoint))]
    async fn post<T, R>(&self, endpoint: &str, request_body: &T) -> Result<R, InferenceClientError>
    where
        T: serde::Serialize + fmt::Debug,
        R: DeserializeOwned + fmt::Debug,
    {
        let url = format!("{}{}", self.base_url, endpoint);
        let span = info_span!("http.request", %url);
        async {
            let response = self
                .client
                .post(&url)
                .json(request_body)
                .send()
                .await
                .map_err(|e| InferenceClientError::Api(e.to_string()))?;

            let status = response.status();
            debug!(%status, "received response");

            if !status.is_success() {
                let error_text = response
                    .text()
                    .await
                    .unwrap_or_else(|_| "Failed to read error body".into());
                error!(%status, body = %error_text, "request failed");
                return Err(InferenceClientError::Api(format!(
                    "Request failed: {status} - {error_text}"
                )));
            }

            let response_text = response.text().await.map_err(|e| {
                InferenceClientError::Api(format!("Failed to read response text: {e}"))
            })?;

            match serde_json::from_str::<R>(&response_text) {
                Ok(parsed) => {
                    trace!(?parsed, "deserialized response");
                    Ok(parsed)
                }
                Err(e) => {
                    error!(%e, raw = %response_text, "deserialization error");
                    Err(InferenceClientError::Serialization(format!(
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
        Pin<Box<dyn Stream<Item = Result<R, InferenceClientError>> + Send + 'static>>,
        InferenceClientError,
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
            .map_err(|e| InferenceClientError::Api(e.to_string()))?;

        if !resp.status().is_success() {
            return Err(InferenceClientError::Api(format!(
                "Request failed: {resp:#?}"
            )));
        }

        let byte_stream = resp.bytes_stream();
        let s = try_stream! {
            let mut buf = Vec::<u8>::new();
            futures::pin_mut!(byte_stream);
            while let Some(chunk) = byte_stream.next().await {
                let chunk = chunk.map_err(|e| InferenceClientError::Request(e.to_string()))?;
                buf.extend_from_slice(&chunk);
                while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
                    let line: Vec<u8> = buf.drain(..=pos).collect();
                    let line = &line[..line.len() - 1];
                    if line.is_empty() { continue; }
                    let parsed: R = serde_json::from_slice(line)
                        .map_err(|e| InferenceClientError::Serialization(e.to_string()))?;
                    yield parsed;
                }
            }
        };
        Ok(Box::pin(s))
    }

    pub async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, InferenceClientError> {
        self.post("/api/chat", &request).await
    }

    pub async fn chat_stream(
        &self,
        req: ChatRequest,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<ChatStreamChunk, InferenceClientError>> + Send + 'static>>,
        InferenceClientError,
    > {
        self.post_stream("/api/chat", &req).await
    }

    pub async fn embeddings(
        &self,
        request: EmbeddingsRequest,
    ) -> Result<EmbeddingsResponse, InferenceClientError> {
        self.post("/api/embeddings", &request).await
    }
}

impl StructuredOuputFormat for OllamaClient {
    fn format(spec: &crate::services::llm::SchemaSpec) -> serde_json::Value {
        spec.schema.clone()
    }
}
