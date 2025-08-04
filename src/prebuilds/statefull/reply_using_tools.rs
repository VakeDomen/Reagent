use crate::{flow::reply_with_tools_flow, flow_types::Flow, prebuilds::StatefullPrebuild, AgentBuilder};

impl StatefullPrebuild {
    pub fn reply_using_tools() -> AgentBuilder {
        AgentBuilder::default()
            .set_flow(Flow::Custom(reply_with_tools_flow))
            .set_name("Statefull-reply_using_tools")
    }
}