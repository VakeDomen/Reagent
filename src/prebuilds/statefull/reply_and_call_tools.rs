use crate::{models::agents::flow::{flows::reply_and_call_tools::reply_and_call_tools_flow, invocation_flows::{Flow, FlowFuture}}, prebuilds::statefull::StatefullPrebuild, util::invocations::invoke, Agent, AgentBuilder, Message};



impl StatefullPrebuild {
    pub fn reply_and_call_tools() -> AgentBuilder {
        AgentBuilder::default()
            .set_flow(Flow::Custom(reply_and_call_tools_flow))
            .set_name("Statefull-reply_and_call_tools")
    }
}