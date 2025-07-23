use crate::{models::agents::flow::invocation_flows::{Flow, FlowFuture}, util::invocations::invoke_without_tools, Agent, AgentBuilder, Message};


fn custom_flow<'a>(agent: &'a mut Agent, prompt: String) -> FlowFuture<'a> {
    Box::pin(async move {
        agent.clear_history();
        agent.history.push(Message::system(agent.system_prompt.clone()));
        agent.history.push(Message::user(prompt));
        let response = invoke_without_tools(agent).await?;
        Ok(response.message)
    })    
}

impl AgentBuilder {
    pub fn reply_without_tools() -> AgentBuilder {
        AgentBuilder::default().set_flow(Flow::Custom(custom_flow))
    }
}