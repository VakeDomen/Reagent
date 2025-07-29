use crate::{models::agents::flow::invocation_flows::{Flow, FlowFuture}, prebuilds::statefull::StatefullPrebuild, util::invocations::invoke_without_tools, Agent, AgentBuilder, Message};


fn custom_flow<'a>(agent: &'a mut Agent, prompt: String) -> FlowFuture<'a> {
    Box::pin(async move {
        agent.history.push(Message::user(prompt));
        let response = invoke_without_tools(agent).await?;
        agent.notify(crate::NotificationContent::Done(true, response.message.content.clone())).await;
        Ok(response.message)
    })    
}

impl StatefullPrebuild {
    pub fn reply_without_tools() -> AgentBuilder {
        AgentBuilder::default()
            .set_flow(Flow::Custom(custom_flow))
            .set_name("Stateless_prebuild-reply_without_tools")
    }
}