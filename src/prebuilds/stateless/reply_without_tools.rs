use crate::{reply_without_tools_flow, Flow, prebuilds::StatelessPrebuild, AgentBuilder};

impl StatelessPrebuild {
    pub fn reply_without_tools() -> AgentBuilder {
        AgentBuilder::default()
            .set_flow(Flow::Custom(reply_without_tools_flow))
            .set_clear_history_on_invocation(true)
            .set_name("Stateless-reply_without_tools")
    }
}