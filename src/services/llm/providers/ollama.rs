use async_stream::try_stream;
use futures::{Stream, StreamExt};
use reqwest::Client;
use serde::de::DeserializeOwned;
use std::{fmt, pin::Pin};
use tracing::{error, span, Instrument, Level, Span};
use tracing_opentelemetry::OpenTelemetrySpanExt;

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

    async fn post<T, R>(&self, endpoint: &str, request_body: &T) -> Result<R, InferenceClientError>
    where
        T: serde::Serialize + fmt::Debug,
        R: DeserializeOwned + fmt::Debug,
    {
        let url = format!("{}{}", self.base_url, endpoint);

        let span = span!(
            Level::INFO,
            "Ollama HTTP Request",
            "langfuse.observation.name" = format!("POST {}", endpoint).as_str(),
            "langfuse.observation.type" = "span",
            "http.request.method" = "POST",
            "url.full" = url.as_str(),
            "server.address" = self.base_url.as_str(),
        );

        if let Ok(body) = serde_json::to_string(request_body) {
            span.set_attribute("langfuse.observation.input", body);
        }

        async {
            let response = self
                .client
                .post(&url)
                .json(request_body)
                .send()
                .await
                .map_err(|e| {
                    Span::current().set_status(opentelemetry::trace::Status::Error {
                        description: e.to_string().into(),
                    });
                    InferenceClientError::Api(e.to_string())
                })?;

            let status = response.status();

            Span::current().set_attribute("http.response.status_code", status.as_u16() as i64);

            if !status.is_success() {
                let error_text = response
                    .text()
                    .await
                    .unwrap_or_else(|_| "Failed to read error body".into());

                error!(%status, body = %error_text, "request failed");

                Span::current().set_status(opentelemetry::trace::Status::Error {
                    description: format!("HTTP {}", status).into(),
                });
                Span::current()
                    .set_attribute("langfuse.observation.status_message", error_text.clone());

                return Err(InferenceClientError::Api(format!(
                    "Ollama request failed: {status} - {error_text}"
                )));
            }

            let response_text = response.text().await.map_err(|e| {
                InferenceClientError::Api(format!("Failed to read response text: {e}"))
            })?;

            Span::current().set_attribute("langfuse.observation.output", response_text.clone());

            match serde_json::from_str::<R>(&response_text) {
                Ok(parsed) => Ok(parsed),
                Err(e) => {
                    error!(%e, raw = %response_text, "deserialization error");
                    Span::current().set_status(opentelemetry::trace::Status::Error {
                        description: "Deserialization Error".into(),
                    });
                    Err(InferenceClientError::Serialization(format!(
                        "Error decoding response body: {e}. Raw JSON was: '{response_text}'"
                    )))
                }
            }
        }
        .instrument(span)
        .await
    }

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
        R: serde::Serialize + serde::de::DeserializeOwned + fmt::Debug + Send + 'static + Clone,
    {
        let url = format!("{}{}", self.base_url, endpoint);

        let span = span!(
            Level::INFO,
            "Ollama HTTP stream",
            "langfuse.observation.name" = format!("POST (Stream) {}", endpoint).as_str(),
            "langfuse.observation.type" = "span",
            "http.request.method" = "POST",
            "url.full" = url.as_str(),
        );

        if let Ok(b) = serde_json::to_string(body) {
            span.set_attribute("langfuse.observation.input", b);
        }

        let stream_span = span.clone();

        let resp = async {
            let resp = self
                .client
                .post(&url)
                .json(body)
                .send()
                .await
                .map_err(|e| {
                    Span::current().set_status(opentelemetry::trace::Status::Error {
                        description: e.to_string().into(),
                    });
                    InferenceClientError::Api(e.to_string())
                })?;

            let status = resp.status();
            Span::current().set_attribute("http.response.status_code", status.as_u16() as i64);

            if !status.is_success() {
                return Err(InferenceClientError::Api(format!("HTTP {}", status)));
            }
            Ok(resp)
        }
        .instrument(span)
        .await?;

        let byte_stream = resp.bytes_stream();

        let s = try_stream! {
            let mut buf = Vec::<u8>::new();
            futures::pin_mut!(byte_stream);

            let mut chunk_count = 0;
            let mut chunks = vec![];

            while let Some(chunk) = byte_stream.next().await {
                let chunk = match chunk {
                    Ok(c) => c,
                    Err(e) => {
                        let err_msg = e.to_string();
                        stream_span.set_attribute("otel.status_code", "ERROR");
                        stream_span.set_attribute("error.message", e.to_string());
                        stream_span.set_status(opentelemetry::trace::Status::Error {
                            description: e.to_string().into(),
                        });
                        Err(InferenceClientError::Request(err_msg))?
                    }
                };


                chunk_count += 1;
                buf.extend_from_slice(&chunk);

                while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
                    let line: Vec<u8> = buf.drain(..=pos).collect();
                    let line = &line[..line.len() - 1];
                    if line.is_empty() { continue; }


                    match serde_json::from_slice::<R>(line) {
                        Ok(parsed) => {
                            chunks.push(parsed.clone());
                            stream_span.set_attribute("langfuse.observation.output", serde_json::to_string_pretty(&chunks).unwrap_or_default());
                            yield parsed
                        },
                        Err(e) => {
                            stream_span.set_attribute("otel.status_code", "ERROR");
                            stream_span.set_attribute("error.message", e.to_string());
                            stream_span.set_status(opentelemetry::trace::Status::Error {
                                description: e.to_string().into(),
                            });
                            Err(InferenceClientError::Serialization(e.to_string()))?;
                        }
                    }
                }
            }

            stream_span.set_attribute("stream.chunk_count", chunk_count);
            stream_span.set_status(opentelemetry::trace::Status::Ok);
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

fn extract_error_telemetry(gen_span: &Span, error_message: &str) {
    gen_span.set_attribute("otel.status_code", "ERROR");
    gen_span.set_attribute("error.message", error_message.to_string());
    gen_span.set_status(opentelemetry::trace::Status::Error {
        description: error_message.to_string().into(),
    });
    error!("stream ended without a final `done` chunk");
}
