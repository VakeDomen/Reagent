use crate::{flow_types::FlowFuture, invocations::invoke_without_tools, Agent, Message};


pub fn reply_without_tools_flow<'a>(agent: &'a mut Agent, prompt: String) -> FlowFuture<'a> {
    Box::pin(async move {
        agent.history.push(Message::user(prompt));
        let response = invoke_without_tools(agent).await?;
        agent.notify(crate::NotificationContent::Done(true, response.message.content.clone())).await;
        Ok(response.message)
    })    
}
