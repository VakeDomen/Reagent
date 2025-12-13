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
//! ```
//! use std::error::Error;
//! use reagent_rs::{init_default_tracing, AgentBuilder};
//! use schemars::{schema_for, JsonSchema};
//! use serde::Deserialize;
//!
//! #[derive(Debug, Deserialize, JsonSchema)]
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
//!         .set_response_format_from::<MyWeatherOuput>()
//!         .set_temperature(0.6)
//!         .set_top_k(20)
//!         .set_stream(true)
//!         .build()
//!         .await?;
//!
//!     let resp: MyWeatherOuput = agent
//!         .invoke_flow_structured_output("What is the current weather in Koper?")
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
pub mod flows;
pub mod notifications;
pub mod observability;
pub mod prebuilds;
pub mod templates;
pub mod tools;

mod services;

pub use crate::agent::*;
pub use crate::flows::*;
pub use crate::notifications::*;
pub use crate::prebuilds::*;
pub use crate::templates::*;
pub use crate::tools::*;

pub use crate::services::llm::{ClientConfig, Provider};

pub use crate::services::llm::models::base::Role;
pub use crate::services::llm::models::chat::{ChatRequest, ChatResponse};
pub use crate::services::llm::models::message::Message;

pub use crate::services::mcp::error::McpIntegrationError;
pub use crate::services::mcp::mcp_tool_builder::McpServerType;

pub use crate::services::logging::init_default_tracing;

pub mod prelude {
    pub use crate::{
        flow, init_default_tracing, Agent, AgentBuildError, AgentBuilder, AgentError, ChatRequest,
        ChatResponse, ClientConfig, Flow, McpIntegrationError, McpServerType, Message,
        Notification, NotificationContent, Provider, Role, Template, TemplateDataSource, Tool,
        ToolBuilder, ToolExecutionError,
    };
}
pub use rmcp::schemars::JsonSchema;
