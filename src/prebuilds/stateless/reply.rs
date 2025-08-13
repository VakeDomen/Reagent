use crate::{reply_flow, Flow, prebuilds::StatelessPrebuild, AgentBuilder};

impl StatelessPrebuild {
    pub fn reply() -> AgentBuilder {
        AgentBuilder::default()
            .set_flow(Flow::Custom(reply_flow))
            .set_clear_history_on_invocation(true)
            .set_name("Stateless-reply_using_tools")
    }
}