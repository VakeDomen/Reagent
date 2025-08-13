use crate::{reply_and_call_tools_flow, Flow, prebuilds::StatelessPrebuild, AgentBuilder};

impl StatelessPrebuild {
    pub fn reply_and_call_tools() -> AgentBuilder {
        AgentBuilder::default()
            .set_flow(Flow::Custom(reply_and_call_tools_flow))
            .set_clear_history_on_invocation(true)
            .set_name("Stateless_prebuild-reply")
    }
}