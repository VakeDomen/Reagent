use crate::{flow, prebuilds::StatefullPrebuild, reply_without_tools_flow, AgentBuilder};

impl StatefullPrebuild {
    pub fn reply_without_tools() -> AgentBuilder {
        AgentBuilder::default()
            .set_flow(flow!(reply_without_tools_flow))
            .set_name("Statefull-reply_without_tools")
    }
}