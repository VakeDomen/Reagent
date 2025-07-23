use crate::{models::agents::flow::invocation_flows::{Flow, FlowFuture}, prebuilds::stateless::StatelessPrebuild, util::invocations::invoke, Agent, AgentBuilder, Message};


fn custom_flow<'a>(agent: &'a mut Agent, prompt: String) -> FlowFuture<'a> {
    Box::pin(async move {
        agent.clear_history();
        agent.history.push(Message::system(agent.system_prompt.clone()));
        agent.history.push(Message::user(prompt));
        let response = invoke(agent).await?;
        Ok(response.message)
    })    
}

impl StatelessPrebuild {
    pub fn reply() -> AgentBuilder {
        AgentBuilder::default()
            .set_flow(Flow::Custom(custom_flow))
            .set_name("Stateless_prebuild-reply")
    }
}