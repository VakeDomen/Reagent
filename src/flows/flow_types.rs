use std::{fmt, future::Future, pin::Pin, sync::Arc};

use crate::{services::ollama::models::chat::ChatResponse, Agent, AgentError, Message};



pub type InvokeFn = for<'a> fn(&'a mut Agent, String) -> InvokeFuture<'a>;
pub type InvokeFuture<'a> = Pin<Box<dyn Future<Output = Result<ChatResponse, AgentError>> + Send + 'a>>;

pub type CustomFlowFn = for<'a> fn(&'a mut Agent, String) -> FlowFuture<'a>;
pub type FlowFn = Arc<dyn for<'a> Fn(&'a mut Agent, String) -> FlowFuture<'a> + Send + Sync>;
pub type FlowFuture<'a> = Pin<Box<dyn Future<Output = Result<Message, AgentError>> + Send + 'a>>;


/// Flow is a user facing enum to
/// define the flow of the agent after 
/// invoking it.
/// The user can choose between predefined 
/// flows or make it's own flow and pass the
/// cusom flow function/closure to the enum
/// variant.
#[derive(Clone)]
pub enum Flow {
    Default,
    Custom(CustomFlowFn),
    CustomClosure(FlowFn),
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


impl Flow {
    pub fn new_closure<F>(f: F) -> Self
    where
        F: for<'a> Fn(&'a mut Agent, String) -> FlowFuture<'a> + Send + Sync + 'static,
    {
        let flow_fn = Arc::new(f);
        Flow::CustomClosure(flow_fn)
    }
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
