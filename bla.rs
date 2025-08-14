// Multi-provider LLM Client Refactor
// =================================
//
// Overview:
// - Introduces a provider-agnostic LLM client trait with a simple factory.
// - Keeps your current message and response models, and generalizes options.
// - Moves the existing Ollama client under providers::ollama and implements the trait.
// - Adds clean error handling and stream support.
// - Stubs for OpenAI, Mistral, Anthropic, and OpenRouter that you can flesh out.
//
// Suggested directory layout:
// src/
//   services/
//     llm/
//       mod.rs
//       client.rs
//       models/
//         mod.rs
//         base.rs
//         chat.rs
//         embedding.rs
//         errors.rs
//       providers/
//         mod.rs
//         ollama.rs
//         openai.rs        // stub
//         mistral.rs       // stub
//         anthropic.rs     // stub
//         openrouter.rs    // stub
//
// Notes:
// - Replace your old `services::ollama` imports with `services::llm`.
// - The generic ChatRequest now carries generalized options; providers translate them.
//

// FILE: src/services/llm/mod.rs


// FILE: src/services/llm/client.rs


// FILE: src/services/llm/models/mod.rs


// FILE: src/services/llm/models/base.rs



// FILE: src/services/llm/models/chat.rs


// FILE: src/services/llm/models/embedding.rs


// FILE: src/services/llm/models/errors.rs


// FILE: src/services/llm/providers/mod.rs


// FILE: src/services/llm/providers/ollama.rs


// FILE: src/services/llm/providers/openai.rs


// FILE: src/services/llm/providers/mistral.rs



// FILE: src/services/llm/providers/anthropic.rs



// FILE: src/services/llm/providers/openrouter.rs
use std::pin::Pin;
use async_trait::async_trait;
use futures::Stream;

use crate::services::llm::client::{ClientConfig, LlmClient};
use crate::services::llm::models::{chat::{ChatRequest, ChatResponse, ChatStreamChunk}, embedding::{EmbeddingsRequest, EmbeddingsResponse}, errors::LlmError};

#[derive(Clone)]
pub struct OpenRouterClient { _cfg: ClientConfig }

impl OpenRouterClient { pub fn new(cfg: ClientConfig) -> Result<Self, LlmError> { Ok(Self { _cfg: cfg }) } }

#[async_trait]
impl LlmClient for OpenRouterClient {
    async fn chat(&self, _req: ChatRequest) -> Result<ChatResponse, LlmError> {
        Err(LlmError::Unsupported("OpenRouter chat not implemented yet".into()))
    }

    async fn chat_stream(
        &self,
        _req: ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatStreamChunk, LlmError>> + Send>>, LlmError> {
        Err(LlmError::Unsupported("OpenRouter streaming not implemented yet".into()))
    }

    async fn embeddings(&self, _req: EmbeddingsRequest) -> Result<EmbeddingsResponse, LlmError> {
        Err(LlmError::Unsupported("OpenRouter embeddings not implemented yet".into()))
    }
}


// FILE: src/examples/llm_chat.rs
// Example usage after the refactor
// cargo run --example llm_chat
use reagent::services::llm::{Client, ClientConfig, Provider};
use reagent::services::llm::models::{Message, Role, chat::ChatRequest, base::{BaseRequest, InferenceOptions}};

#[tokio::main]
async fn main() {
    let client = Client::new(ClientConfig {
        provider: Provider::Ollama,
        base_url: Some("http://localhost:11434".into()),
        ..Default::default()
    }).expect("client");

    let req = ChatRequest {
        base: BaseRequest {
            model: "llama3.1".into(),
            format: None,
            options: Some(InferenceOptions { temperature: Some(0.2), ..Default::default() }),
            stream: Some(false),
            keep_alive: None,
        },
        messages: vec![
            Message::system("You are a helpful assistant."),
            Message::user("Hi, who are you?"),
        ],
        tools: None,
    };

    let resp = client.chat(req).await.expect("chat");
    println!("{}", resp.message.content.unwrap_or_default());
}
