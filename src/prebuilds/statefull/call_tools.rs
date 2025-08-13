use crate::{prebuilds::StatefullPrebuild, call_tools_flow, AgentBuilder, Flow};

impl StatefullPrebuild {
    pub fn call_tools() -> AgentBuilder {
        AgentBuilder::default()
            .set_flow(Flow::Custom(call_tools_flow))
            .set_name("Statefull-reply_and_call_tools")
    }
}