use reqwest::{Client, Error as ReqwestError};
use serde::de::DeserializeOwned;
use std::fmt;

use super::models::{ChatRequest, ChatResponse, EmbeddingsRequest, EmbeddingsResponse, GenerateRequest, GenerateResponse, OllamaError};

/// The main client for interacting with the Ollama API.
#[derive(Debug, Clone)]
pub struct OllamaClient {
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
    async fn post<T, R>(&self, endpoint: &str, request_body: &T) -> Result<R, OllamaError>
    where
        T: serde::Serialize + fmt::Debug,
        R: DeserializeOwned + fmt::Debug,
    {
        let url = format!("{}{}", self.base_url, endpoint);
        println!("Sending request to: {}", url);
        println!("Request body: {:?}", serde_json::to_string(request_body));

        let response = match self
            .client
            .post(&url)
            .json(request_body)
            .send()
            .await {
                Ok(r) => r,
                Err(e) => return Err(OllamaError::ApiError(e.to_string())),
            };

        if response.status().is_success() {
            let result = match response.json::<R>().await {
                Ok(r) => r,
                Err(e) => return Err(OllamaError::SerializationError(e.to_string())),
            };
            println!("Received response: {:?}", result);
            Ok(result)
        } else {
            let status = response.status();
            let text = response.text().await.unwrap_or_else(|_| "Failed to read error body".to_string());
            println!("Request failed with status: {}", status);
            println!("Error body: {}", text);
            Err(OllamaError::ApiError(format!(
                "Request failed: {} - {}",
                status, text
            )))
        }
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
        println!("Sending GET request to: {}", url);
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