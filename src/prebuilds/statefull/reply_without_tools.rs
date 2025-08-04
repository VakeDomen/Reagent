use crate::{models::agents::flow::{flows::reply_without_tools::reply_without_tools_flow, invocation_flows::{Flow, FlowFuture}}, prebuilds::statefull::StatefullPrebuild, util::invocations::invoke_without_tools, Agent, AgentBuilder, Message};


impl StatefullPrebuild {
    pub fn reply_without_tools() -> AgentBuilder {
        AgentBuilder::default()
            .set_flow(Flow::Custom(reply_without_tools_flow))
            .set_name("Statefull-reply_without_tools")
    }
}