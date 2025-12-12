mod call_tools;
mod default_flow;
mod flow_types;
mod reply;
mod reply_without_tools;

pub use self::{
    call_tools::call_tools_flow, default_flow::default_flow, flow_types::*, reply::reply_flow,
    reply_without_tools::reply_without_tools_flow,
};

#[macro_export]
macro_rules! flow {
    ($f:expr) => {
        |agent: &mut $crate::Agent, prompt: ::std::string::String| -> $crate::FlowFuture<'_> {
            ::std::boxed::Box::pin($f(agent, prompt))
        }
    };
}
