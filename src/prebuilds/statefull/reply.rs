use crate::{flow, prebuilds::StatefullPrebuild, reply_flow, AgentBuilder};

impl StatefullPrebuild {
    pub fn reply() -> AgentBuilder {
        AgentBuilder::default()
            .set_flow(flow!(reply_flow))
            .set_name("Statefull-reply_using_tools")
    }
}