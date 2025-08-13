use crate::{prebuilds::StatefullPrebuild, reply_and_call_tools_flow, AgentBuilder, Flow};

impl StatefullPrebuild {
    pub fn reply_and_call_tools() -> AgentBuilder {
        AgentBuilder::default()
            .set_flow(Flow::Custom(reply_and_call_tools_flow))
            .set_name("Statefull-reply_and_call_tools")
    }
}