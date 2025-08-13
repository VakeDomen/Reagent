use crate::{call_tools_flow, Flow, prebuilds::StatelessPrebuild, AgentBuilder};

impl StatelessPrebuild {
    pub fn call_tools() -> AgentBuilder {
        AgentBuilder::default()
            .set_flow(Flow::Custom(call_tools_flow))
            .set_clear_history_on_invocation(true)
            .set_name("Stateless_prebuild-reply")
    }
}