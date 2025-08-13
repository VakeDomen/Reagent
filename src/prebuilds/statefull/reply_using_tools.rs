use crate::{reply_with_tools_flow, Flow, prebuilds::StatefullPrebuild, AgentBuilder};

impl StatefullPrebuild {
    pub fn reply_using_tools() -> AgentBuilder {
        AgentBuilder::default()
            .set_flow(Flow::Custom(reply_with_tools_flow))
            .set_name("Statefull-reply_using_tools")
    }
}