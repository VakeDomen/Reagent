use crate::{models::agents::flow::{flows::reply_with_tools::reply_with_tools_flow, invocation_flows::{Flow, FlowFuture}}, prebuilds::stateless::StatelessPrebuild, util::invocations::{call_tools, invoke}, Agent, AgentBuilder, Message};

impl StatelessPrebuild {
    pub fn reply_using_tools() -> AgentBuilder {
        AgentBuilder::default()
            .set_flow(Flow::Custom(reply_with_tools_flow))
            .set_clear_history_on_invocation(true)
            .set_name("Stateless-reply_using_tools")
    }
}