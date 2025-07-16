use reqwest::{Client, Error as ReqwestError};
use serde::de::DeserializeOwned;
use tracing::{debug, error, info_span, instrument, Instrument};
use std::fmt;

use super::models::{chat::{ChatRequest, ChatResponse}, embedding::{EmbeddingsRequest, EmbeddingsResponse}, errors::OllamaError, generate::{GenerateRequest, GenerateResponse}};


/// The main client for interacting with the Ollama API.
#[derive(Debug, Clone)]
pub(crate) struct OllamaClient {
    client: Client,
    base_url: String,
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
                .map_err(|e| OllamaError::ApiError(e.to_string()))?;

            let status = response.status();
            debug!(%status, "received response");

            // Successful status path
            if status.is_success() {
                let response_text = response
                    .text()
                    .await
                    .map_err(|e| OllamaError::ApiError(format!("Failed to read response text: {e}")))?;

                match serde_json::from_str::<R>(&response_text) {
                    Ok(parsed) => {
                        debug!(?parsed, "deserialized response");
                        Ok(parsed)
                    }
                    Err(e) => {
                        error!(%e, raw = %response_text, "deserialization error");
                        Err(OllamaError::SerializationError(format!(
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
                Err(OllamaError::ApiError(format!(
                    "Request failed: {status} - {error_text}"
                )))
            }
        }
        .instrument(span)
        .await
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
            Err(e) => Err(OllamaError::RequestError(e.to_string())),
        }
    }
}

impl From<ReqwestError> for OllamaError {
    fn from(err: ReqwestError) -> Self {
        OllamaError::RequestError(err.to_string())
    }
}