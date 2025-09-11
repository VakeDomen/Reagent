use crate::{flow, prebuilds::StatelessPrebuild, reply_flow, AgentBuilder};

impl StatelessPrebuild {
    pub fn reply() -> AgentBuilder {
        AgentBuilder::default()
            .set_flow(flow!(reply_flow))
            .set_clear_history_on_invocation(true)
            .set_name("Stateless-reply_using_tools")
    }
}