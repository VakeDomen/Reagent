use async_stream::try_stream;
use futures::Stream;
use reqwest::{Client, Error as ReqwestError};
use serde::de::DeserializeOwned;
use futures::StreamExt;
use tracing::{debug, error, info_span, trace, instrument, Instrument};
use std::fmt;

use crate::services::ollama::models::{chat::ChatStreamChunk, generate::GenerateStreamChunk};

use super::models::{chat::{ChatRequest, ChatResponse}, embedding::{EmbeddingsRequest, EmbeddingsResponse}, errors::OllamaError, generate::{GenerateRequest, GenerateResponse}};


/// The main client for interacting with the Ollama API.
#[derive(Debug, Clone)]
pub(crate) struct OllamaClient {
    pub client: Client,
    pub base_url: String,
}

impl OllamaClient {
    /// Creates a new `OllamaClient`.
    ///
    /// # Arguments
    ///
    /// * `base_url` - The base URL of the Ollama API (e.g., "http://localhost:11434").
    pub fn new(base_url: String) -> Self {
        OllamaClient {
            client: Client::new(),
            base_url,
        }
    }

    /// Creates a new `OllamaClient` with the default base URL ("http://localhost:11434").
    pub fn default() -> Self {
        Self::new("http://localhost:11434".to_string())
    }

    /// Executes a POST request to the specified Ollama API endpoint.
    ///
    /// # Arguments
    ///
    /// * `endpoint` - The API endpoint path (e.g., "/api/generate").
    /// * `request_body` - The request payload, which will be serialized to JSON.
    ///
    /// # Returns
    ///
    /// A `Result` containing the deserialized response or an `OllamaError`.
    /// 
    #[instrument(
        name = "ollama.post",
        skip_all,
        fields(
            endpoint,
        )
    )]
    async fn post<T, R>(&self, endpoint: &str, request_body: &T) -> Result<R, OllamaError>
    where
        T: serde::Serialize + fmt::Debug,
        R: DeserializeOwned + fmt::Debug,
    {
        // Build full URL once so we can record it
        let url = format!("{}{}", self.base_url, endpoint);

        // Attach a child span for the HTTP call itself
        let span = info_span!("http.request", %url);
        async {
            // debug!("{} {:?}", "sending request", serde_json::to_string(request_body));

            // Perform the POST
            let response = self
                .client
                .post(&url)
                .json(request_body)
                .send()
                .await
                .map_err(|e| OllamaError::Api(e.to_string()))?;

            let status = response.status();
            debug!(%status, "received response");

            // Successful status path
            if status.is_success() {
                let response_text = response
                    .text()
                    .await
                    .map_err(|e| OllamaError::Api(format!("Failed to read response text: {e}")))?;

                match serde_json::from_str::<R>(&response_text) {
                    Ok(parsed) => {
                        trace!(?parsed, "deserialized response");
                        Ok(parsed)
                    }
                    Err(e) => {
                        error!(%e, raw = %response_text, "deserialization error");
                        Err(OllamaError::Serialization(format!(
                            "Error decoding response body: {e}. Raw JSON was: '{response_text}'"
                        )))
                    }
                }
            } else {
                let error_text = response
                    .text()
                    .await
                    .unwrap_or_else(|_| "Failed to read error body".into());

                error!(%status, body = %error_text, "request failed");
                Err(OllamaError::Api(format!(
                    "Request failed: {status} - {error_text}"
                )))
            }
        }
        .instrument(span)
        .await
    }



    #[instrument(
        name = "ollama.post_stream",
        skip_all,
        fields(
            endpoint,
        )
    )]
    async fn post_stream<T, R>(
        &self,
        endpoint: &str,
        body: &T,
    ) -> Result<impl Stream<Item = Result<R, OllamaError>> + Send + 'static, OllamaError>
    where
        T: serde::Serialize + fmt::Debug,
        R: serde::de::DeserializeOwned + fmt::Debug + Send + 'static,
    {
        let url  = format!("{}{}", self.base_url, endpoint);
        let resp = self.client.post(&url).json(body).send().await
            .map_err(|e| OllamaError::Api(e.to_string()))?;

        if !resp.status().is_success() {
            return Err(OllamaError::Api(format!("Request failed: {:#?}", resp)));
        }

        let byte_stream = resp.bytes_stream();

        let s = try_stream! {
            let mut buf = Vec::<u8>::new();
            tokio::pin!(byte_stream);

            while let Some(chunk) = byte_stream.next().await {
                let chunk = chunk.map_err(|e| OllamaError::Request(e.to_string()))?;
                buf.extend_from_slice(&chunk);

                // split on LF – Ollama always sends \n-terminated JSON lines
                while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
                    let line: Vec<u8> = buf.drain(..=pos).collect();
                    let line = &line[..line.len() - 1]; // trim LF
                    if line.is_empty() { continue; }    // heartbeat newline

                    let parsed: R = serde_json::from_slice(line)
                        .map_err(|e| OllamaError::Serialization(e.to_string()))?;
                    yield parsed;
                }
            }
        };

        Ok(s)
    }

    /// Sends a generation request to the Ollama API.
    ///
    /// # Arguments
    ///
    /// * `request` - The `GenerateRequest` containing the model, prompt, and options.
    ///
    /// # Returns
    ///
    /// A `Result` with the `GenerateResponse` or an `OllamaError`.
    pub async fn generate(
        &self,
        request: GenerateRequest,
    ) -> Result<GenerateResponse, OllamaError> {
        self.post("/api/generate", &request).await
    }

    /// Sends a chat request to the Ollama API.
    ///
    /// # Arguments
    ///
    /// * `request` - The `ChatRequest` containing the model, messages, tools, and options.
    ///
    /// # Returns
    ///
    /// A `Result` with the `ChatResponse` or an `OllamaError`.
    pub async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, OllamaError> {
        self.post("/api/chat", &request).await
    }

    pub async fn generate_stream(
        &self,
        mut req: GenerateRequest,
    ) -> Result<impl Stream<Item = Result<GenerateStreamChunk, OllamaError>> + Send + 'static, OllamaError>
    {
        req.base.stream = Some(true);
        self.post_stream("/api/generate", &req).await
    }

    pub async fn chat_stream(
        &self,
        mut req: ChatRequest,
    ) -> Result<impl Stream<Item = Result<ChatStreamChunk, OllamaError>> + Send + 'static, OllamaError>
    {
        req.base.stream = Some(true);
        self.post_stream("/api/chat", &req).await
    }



    /// Generates embeddings for a given prompt using the specified model.
    /// Corresponds to the `/api/embeddings` endpoint.
    ///
    /// # Arguments
    ///
    /// * `request` - The `EmbeddingsRequest` containing the model and prompt.
    ///
    /// # Returns
    ///
    /// A `Result` with the `EmbeddingsResponse` or an `OllamaError`.
    pub async fn embeddings(
        &self,
        request: EmbeddingsRequest,
    ) -> Result<EmbeddingsResponse, OllamaError> {
        self.post("/api/embeddings", &request).await
    }


    /// Checks if the Ollama server is running.
    /// Corresponds to the `HEAD /` endpoint (we use GET for simplicity).
    ///
    /// # Returns
    ///
    /// A `Result` containing `true` if the server is up, or an `OllamaError`.
    pub async fn heartbeat(&self) -> Result<bool, OllamaError> {
        let url = &self.base_url;
        match self.client.get(url).send().await {
            Ok(response) => Ok(response.status().is_success()),
            Err(e) => Err(OllamaError::Request(e.to_string())),
        }
    }
}

impl From<ReqwestError> for OllamaError {
    fn from(err: ReqwestError) -> Self {
        OllamaError::Request(err.to_string())
    }
}