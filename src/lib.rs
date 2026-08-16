//! You can import everything directly from the crate:
//! ```rust
//! use reagent_rs::{Agent, AgentBuilder, Flow, Tool, Message};
//! ```
//! Or pull in the essentials:
//! ```rust
//! use reagent_rs::prelude::*;
//! ```
//!
//! Create an [`Agent`] using [`AgentBuilder`] :
//!
//! ```no_run
//! use std::error::Error;
//! use reagent_rs::{init_default_tracing, AgentBuilder};
//! use schemars::JsonSchema;
//! use serde::{Deserialize, Serialize};
//!
//! #[derive(Debug, Deserialize, Serialize, JsonSchema)]
//! struct MyWeatherOuput {
//!   windy: bool,
//!   temperature: i32,
//!   description: String
//! }
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn Error>> {
//!     init_default_tracing();
//!
//!     let mut agent = AgentBuilder::default()
//!         .set_model("qwen3:0.6b")
//!         .set_system_prompt("You make up weather info in JSON")
//!         .structured_output::<MyWeatherOuput>()
//!         .set_temperature(0.6)
//!         .set_top_k(20)
//!         .set_stream(true)
//!         .build()
//!         .await?;
//!
//!     let resp = agent
//!         .invoke("What is the current weather in Koper?")
//!         .await?;
//!
//!     Ok(())
//! }
//! ```
//!
//!
//! Reagent talks to Ollama by default. It also supports OpenRouter.
//! To use OpenRouter, set the provider to `Provider::OpenRouter` and supply your API key.
//!
//! ```rust
//!
//! use reagent_rs::{AgentBuilder, Provider};
//!
//! async {
//!     let agent = AgentBuilder::default()
//!         .set_provider(Provider::OpenRouter)
//!         .set_api_key("your_openrouter_key")
//!         .set_model("meta-llama/llama-3.1-8b-instruct:free")
//!         .build()
//!         .await;
//! };
//! ```

#![forbid(unsafe_code)]

pub mod agent;
pub mod invocation;
pub mod model;
pub mod notifications;
pub mod observability;
pub mod skills;
pub mod templates;
pub mod tools;

mod services;

pub use crate::agent::*;
pub use crate::invocation::{
    ChatInvocation, EmbeddingInvocation, Invocation, InvocationError, Standard, Structured,
};
pub use crate::model::*;
pub use crate::notifications::*;
pub use crate::skills::*;
pub use crate::templates::*;
pub use crate::tools::*;

pub use crate::observability::init_default_tracing;
pub use crate::services::llm::{ClientConfig, InferenceOptions, Provider, SchemaSpec};

pub use crate::services::llm::models::base::Role;
pub use crate::services::llm::models::chat::{ChatRequest, ChatResponse};
pub use crate::services::llm::models::embedding::EmbeddingsResponse;
pub use crate::services::llm::models::message::Message;

pub use crate::services::mcp::error::McpIntegrationError;
pub use crate::services::mcp::mcp_tool_builder::McpServerType;

pub mod prelude {
    pub use crate::{
        flow, Agent, AgentBuildError, AgentBuilder, AgentError, ChatInvocation, ChatRequest,
        ChatResponse, ClientConfig, Embedding, EmbeddingInvocation, EmbeddingModel,
        EmbeddingModelBuilder, EmbeddingsResponse, Flow, InferenceOptions, Invocation, Llm,
        LlmModel, LlmModelBuilder, LoadTemplateError, McpIntegrationError, McpServerType, Message,
        Model, ModelBuilder, Notification, NotificationContent, Prompt, Provider, Role, SchemaSpec,
        Skill, SkillLoadError, SkillResource, SkillResourceKind, Standard, Structured, Template,
        TemplateDataSource, TemplateInput, Tool, ToolBuilder, ToolExecutionError,
    };
}

pub use rmcp::schemars::JsonSchema;
pub use serde_json::Value;
