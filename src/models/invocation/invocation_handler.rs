use std::{fmt, future::Future, pin::Pin, sync::Arc};

use crate::{models::AgentError, services::ollama::models::chat::ChatResponse, Agent, Message};

pub type CustomFlowFn = for<'a> fn(&'a mut Agent, String) -> FlowFuture<'a>;

/// The public-facing enum that users interact with via the `AgentBuilder`.
#[derive(Clone, Debug)]
pub enum Flow {
    Simple,
    /// A custom flow, provided as a function pointer.
    /// It must match the `CustomFlowFn` signature.
    Custom(CustomFlowFn),
}


impl From<Flow> for InternalFlow {
    fn from(flow: Flow) -> Self {
        match flow {
            Flow::Simple => InternalFlow::Simple,
            Flow::Custom(custom_fn_ptr) => {
                // The function pointer `custom_fn_ptr` is wrapped in an Arc.
                // Rust can automatically coerce a `fn` pointer into a `dyn Fn` trait object.
                // This creates our clonable, type-erased `FlowFn`.
                InternalFlow::Custom(Arc::new(custom_fn_ptr))
            }
        }
    }
}


// #[derive(Clone)]
// pub enum InternalFlow {
//   Simple,
//   Custom(FlowFn),
// }

// impl fmt::Debug for InternalFlow {
//   fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//     match self {
//       InternalFlow::Simple    => f.write_str("InternalFlow::Simple"),
//       InternalFlow::Custom(_) => f.write_str("InternalFlow::Custom(<flow>)"),
//     }
//   }
// }

// pub type FlowFn = Arc<dyn for<'a> Fn(&'a mut Agent, String) -> FlowFuture<'a> + Send + Sync>;

/// The return type of any flow function: a pinned, boxed, dynamic Future.
/// This is the "type-erased" future.
// pub type FlowFuture<'a> = Pin<Box<dyn Future<Output = Result<Message, AgentError>> + Send + 'a>>;

/// The internal representation of a flow stored within the Agent.
#[derive(Clone)]
pub enum InternalFlow {
  Simple,
  Custom(FlowFn),
}

impl fmt::Debug for InternalFlow {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      InternalFlow::Simple    => f.write_str("InternalFlow::Simple"),
      // We can't inspect the function, so we just label it.
      InternalFlow::Custom(_) => f.write_str("InternalFlow::Custom(<flow_fn>)"),
    }
  }
}


pub type InvokeFn = for<'a> fn(&'a mut Agent, String) -> InvokeFuture<'a>;
pub type InvokeFuture<'a> = Pin<Box<dyn Future<Output = Result<ChatResponse, AgentError>> + Send + 'a>>;

pub type FlowFn = Arc<dyn for<'a> Fn(&'a mut Agent, String) -> FlowFuture<'a> + Send + Sync>;
pub type FlowFuture<'a> = Pin<Box<dyn Future<Output = Result<Message, AgentError>> + Send + 'a>>;

