use crate::{flow, prebuilds::StatelessPrebuild, reply_without_tools_flow, AgentBuilder};

impl StatelessPrebuild {
    pub fn reply_without_tools() -> AgentBuilder {
        AgentBuilder::default()
            .set_flow(flow!(reply_without_tools_flow))
            .set_clear_history_on_invocation(true)
            .remove_tools()
            .set_name("Stateless-reply_without_tools")
    }
}
