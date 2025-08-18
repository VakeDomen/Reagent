use std::{fmt, future::Future, pin::Pin, sync::Arc};

use crate::{services::llm::models::chat::ChatResponse, Agent, AgentError, Message};



pub type InvokeFn = for<'a> fn(&'a mut Agent, String) -> InvokeFuture<'a>;
pub type InvokeFuture<'a> = Pin<Box<dyn Future<Output = Result<ChatResponse, AgentError>> + Send + 'a>>;

pub type CustomFlowFn = for<'a> fn(&'a mut Agent, String) -> FlowFuture<'a>;
pub type ClosureFlowFn = Arc<dyn for<'a> Fn(&'a mut Agent, String) -> FlowFuture<'a> + Send + Sync>;
pub type FlowFn = Arc<dyn for<'a> Fn(&'a mut Agent, String) -> FlowFuture<'a> + Send + Sync>;
pub type FlowFuture<'a> = Pin<Box<dyn Future<Output = Result<Message, AgentError>> + Send + 'a>>;


/// A user-facing enum defining how an [`Agent`] executes a flow
/// after receiving a prompt.
///
/// The flow determines how prompts are handled and how responses
/// are generated. By default, the agent uses the built-in flow,
/// but you can(and should) also provide custom functions or closures.
///
/// # Variants
/// - [`Flow::Default`] — use the built-in default flow.
/// - [`Flow::Custom`] — supply a function pointer with the correct signature.
/// - [`Flow::CustomClosure`] — supply a closure wrapped in an `Arc`.
#[derive(Clone)]
pub enum Flow {
    /// Use the built-in default flow.
    Default,
    /// Use a custom function pointer.
    ///
    /// Function must match `for<'a> fn(&'a mut Agent, String) -> FlowFuture<'a>`.
    Custom(CustomFlowFn),
    /// Use a custom closure.
    ///
    /// Closure must be `Send + Sync + 'static` and match
    /// `for<'a> Fn(&'a mut Agent, String) -> FlowFuture<'a>`.
    CustomClosure(ClosureFlowFn),
}





impl Flow {
    /// Create a new [`Flow::CustomClosure`] from a closure.
    ///
    /// This is the more ergonomic way to supply a custom flow closure,
    /// since it accepts any closure or function that matches the
    /// required signature and wraps it in an `Arc`.
    pub fn new_closure<F>(f: F) -> Self
    where
        F: for<'a> Fn(&'a mut Agent, String) -> FlowFuture<'a> + Send + Sync + 'static,
    {
        let flow_fn = Arc::new(f);
        Flow::CustomClosure(flow_fn)
    }
}


/// InternalFlow is a translated version
/// of the Flow enum, intended for internal
/// use of the library. It wrapps/translates
/// the user defined flows to versions 
/// understandable to the library.
#[derive(Clone)]
pub(crate) enum InternalFlow {
    Default,
    Custom(FlowFn),
}

impl From<Flow> for InternalFlow {
    fn from(flow: Flow) -> Self {
        match flow {
            Flow::Default => InternalFlow::Default,
            Flow::Custom(custom_fn_ptr) => {
                InternalFlow::Custom(Arc::new(custom_fn_ptr))
            }
            Flow::CustomClosure(flow_fn) => InternalFlow::Custom(flow_fn),
        }
    }
}






// ------------ custom debugs ------------ 


impl fmt::Debug for InternalFlow {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InternalFlow::Default    => f.write_str("InternalFlow::Default"),
            InternalFlow::Custom(_) => f.write_str("InternalFlow::Custom(<flow_fn>)"),
        }
  }
}

impl fmt::Debug for Flow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Flow::Default => write!(f, "Simple"),
            Flow::Custom(func_ptr) => f.debug_tuple("Custom").field(func_ptr).finish(),
            Flow::CustomClosure(_) => write!(f, "CustomClosure(<closure>)"),
        }
    }
}
