use crate::{models::agents::flow::{flows::reply_without_tools::reply_without_tools_flow, invocation_flows::{Flow, FlowFuture}}, prebuilds::stateless::StatelessPrebuild, util::invocations::invoke_without_tools, Agent, AgentBuilder, Message};

impl StatelessPrebuild {
    pub fn reply_without_tools() -> AgentBuilder {
        AgentBuilder::default()
            .set_flow(Flow::Custom(reply_without_tools_flow))
            .set_clear_history_on_invocation(true)
            .set_name("Stateless-reply_without_tools")
    }
}