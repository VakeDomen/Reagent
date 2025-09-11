use crate::{call_tools_flow, flow, prebuilds::StatefullPrebuild, AgentBuilder};

impl StatefullPrebuild {
    pub fn call_tools() -> AgentBuilder {
        AgentBuilder::default()
            .set_flow(flow!(call_tools_flow))
            .set_name("Statefull-reply_and_call_tools")
    }
}