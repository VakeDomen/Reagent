use std::{future::Future, pin::Pin};

use crate::{models::AgentError, services::ollama::models::chat::ChatResponse, Agent, Message};

pub type InvokeFn = for<'a> fn(&'a mut Agent, String) -> InvokeFuture<'a>;
pub type InvokeFuture<'a> = Pin<Box<dyn Future<Output = Result<ChatResponse, AgentError>> + Send + 'a>>;

pub type FlowFn = for<'a> fn(&'a mut Agent, String) -> FlowFuture<'a>;
pub type FlowFuture<'a> = Pin<Box<dyn Future<Output = Result<Message, AgentError>> + Send + 'a>>;

