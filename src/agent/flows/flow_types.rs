use std::{fmt, future::Future, pin::Pin, sync::Arc};

use crate::{services::llm::message::Message, Agent, AgentError, Prompt, Standard};

pub type FlowFuture<'a> = Pin<Box<dyn Future<Output = Result<Message, AgentError>> + Send + 'a>>;

pub type FlowFn<I = Prompt, O = Standard> =
    Arc<dyn for<'a> Fn(&'a mut Agent<I, O>, String) -> FlowFuture<'a> + Send + Sync>;

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
pub enum Flow<I = Prompt, O = Standard> {
    /// Use the built-in default flow.
    Default,
    /// Use a custom function pointer.
    ///
    /// Function must match `for<'a> fn(&'a mut Agent, String) -> FlowFuture<'a>`.
    Func(FlowFn<I, O>),
}

impl<I, O> Clone for Flow<I, O> {
    fn clone(&self) -> Self {
        match self {
            Self::Default => Self::Default,
            Self::Func(flow) => Self::Func(flow.clone()),
        }
    }
}

impl<I, O> Flow<I, O> {
    pub fn from_fn<F>(f: F) -> Self
    where
        F: for<'a> Fn(&'a mut Agent<I, O>, String) -> FlowFuture<'a> + Send + Sync + 'static,
    {
        Flow::Func(Arc::new(f))
    }
}

// ------------ custom debugs ------------

impl<I, O> fmt::Debug for Flow<I, O> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Flow::Default => write!(f, "Simple"),
            Flow::Func(_) => write!(f, "CustomFlow(<fn>)"),
        }
    }
}

pub trait FlowCallable: Send + Sync + 'static {
    // family of futures tied to the borrow of &mut Agent
    type Fut<'a>: Future<Output = Result<Message, AgentError>> + Send + 'a
    where
        Self: 'a;

    fn call<'a>(&'a self, agent: &'a mut Agent, prompt: String) -> Self::Fut<'a>;
}
