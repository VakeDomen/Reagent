use crate::{models::agents::flow::{flows::reply_with_tools::reply_with_tools_flow, invocation_flows::{Flow, FlowFuture}}, prebuilds::statefull::StatefullPrebuild, util::invocations::{call_tools, invoke}, Agent, AgentBuilder, Message};



impl StatefullPrebuild {
    pub fn reply_using_tools() -> AgentBuilder {
        AgentBuilder::default()
            .set_flow(Flow::Custom(reply_with_tools_flow))
            .set_name("Statefull-reply_using_tools")
    }
}