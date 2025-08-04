use crate::{flow::reply_without_tools_flow, flow_types::Flow, prebuilds::StatefullPrebuild, AgentBuilder};

impl StatefullPrebuild {
    pub fn reply_without_tools() -> AgentBuilder {
        AgentBuilder::default()
            .set_flow(Flow::Custom(reply_without_tools_flow))
            .set_name("Statefull-reply_without_tools")
    }
}