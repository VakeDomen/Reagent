use crate::{flow::reply_and_call_tools_flow, flow_types::Flow, prebuilds::StatefullPrebuild, AgentBuilder};

impl StatefullPrebuild {
    pub fn reply_and_call_tools() -> AgentBuilder {
        AgentBuilder::default()
            .set_flow(Flow::Custom(reply_and_call_tools_flow))
            .set_name("Statefull-reply_and_call_tools")
    }
}