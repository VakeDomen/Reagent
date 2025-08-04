use crate::{models::agents::flow::{flows::reply_and_call_tools::reply_and_call_tools_flow, invocation_flows::Flow}, prebuilds::stateless::StatelessPrebuild, util::invocations::invoke, Agent, AgentBuilder, Message};


impl StatelessPrebuild {
    pub fn reply_and_call_tools() -> AgentBuilder {
        AgentBuilder::default()
            .set_flow(Flow::Custom(reply_and_call_tools_flow))
            .set_clear_history_on_invocation(true)
            .set_name("Stateless_prebuild-reply")
    }
}