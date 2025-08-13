use crate::{reply_flow, Flow, prebuilds::StatefullPrebuild, AgentBuilder};

impl StatefullPrebuild {
    pub fn reply() -> AgentBuilder {
        AgentBuilder::default()
            .set_flow(Flow::Custom(reply_flow))
            .set_name("Statefull-reply_using_tools")
    }
}