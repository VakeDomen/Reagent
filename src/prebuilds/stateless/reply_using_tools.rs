use crate::{flow::reply_with_tools_flow, flow_types::Flow, prebuilds::StatelessPrebuild, AgentBuilder};

impl StatelessPrebuild {
    pub fn reply_using_tools() -> AgentBuilder {
        AgentBuilder::default()
            .set_flow(Flow::Custom(reply_with_tools_flow))
            .set_clear_history_on_invocation(true)
            .set_name("Stateless-reply_using_tools")
    }
}