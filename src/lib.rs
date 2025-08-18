//! Reagent crate root
//!
//! You can import everything directly from the crate:
//! ```rust
//! use reagent::{Agent, AgentBuilder, Flow, Tool, Message};
//! ```
//! Or pull in the essentials:
//! ```rust
//! use reagent::prelude::*;
//! ```
//! 
//! Create [`Agent`] using [`AgentBuilder`] :
//! 
//! ```
//! use std::error::Error;
//! use reagent::{init_default_tracing, AgentBuilder};
//! 
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn Error>> {
//!     init_default_tracing();
//!     
//!     // creating agents follows the builder pattern
//!     let mut agent = AgentBuilder::default()
//!         // model must be set, everything else has 
//!         // defualts and is optional
//!         .set_model("qwen3:0.6b")
//!         .set_system_prompt("You are a helpful assistant.")
//!         .set_temperature(0.6)
//!         .set_num_ctx(2048) // lol
//!         // call build to return the agent
//!         .build()
//!         // creation can fail (sever unreachable?)
//!         .await?;
//! 
//!     // call agents by calling the "invoke_flow" method
//!     let resp = agent.invoke_flow("How do i increase context size in Ollama?").await?;
//!     println!("\n-> Agent: {}", resp.content.unwrap_or_default());
//! 
//!     // internally agent holds the conversation histroy
//!     let resp = agent.invoke_flow("What did you just say?").await?;
//!     println!("\n-> Agent: {}", resp.content.unwrap_or_default());
//! 
//!     // but it can be reset
//!     // system message will stay, other messages will
//!     // be deleted
//!     agent.clear_history();
//! 
//!     let resp = agent.invoke_flow("What did you just say?").await?;
//!     println!("\n-> Agent: {}", resp.content.unwrap_or_default());
//! 
//! 
//!     Ok(())
//! }
//! ```


#![forbid(unsafe_code)]

pub mod agent;
pub mod flows;
pub mod notifications;
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

pub use crate::services::llm::models::base::{Message, Role};
pub use crate::services::llm::models::chat::{ChatRequest, ChatResponse};

pub use crate::services::mcp::mcp_tool_builder::McpServerType;
pub use crate::services::mcp::error::McpIntegrationError;

pub use crate::services::logging::init_default_tracing;

// Invocation helpers at top level for convenience
pub mod invocations {
    pub use crate::agent::{
        call_tools as invoke_call_tools,
        invoke,
        invoke_with_tool_calls,
        invoke_without_tools,
    };
}

// -------------------------------------
// Small prelude for common imports
// -------------------------------------

/// Commonly used items for building and running agents.
pub mod prelude {
    pub use crate::{
        // Core
        Agent, AgentBuilder, AgentBuildError, AgentError,
        // Flows
        Flow,
        // Tools
        Tool, ToolBuilder, ToolExecutionError,
        // Notifications
        Notification, NotificationContent,
        // Templates
        Template, TemplateDataSource,
        // LLM
        ClientConfig, Provider, Message, Role, ChatRequest, ChatResponse,
        // MCP
        McpServerType, McpIntegrationError,
        // Logging
        init_default_tracing,
        // Invocation helpers
        invoke, invoke_with_tool_calls, invoke_without_tools,
    };
}
