use crate::{models::agents::flow::invocation_flows::FlowFuture, util::invocations::invoke, Agent, Message};

pub fn reply_with_tools_flow<'a>(agent: &'a mut Agent, prompt: String) -> FlowFuture<'a> {
    Box::pin(async move {
        agent.history.push(Message::user(prompt));
        let response = invoke(agent).await?;
        agent.notify(crate::NotificationContent::Done(true, response.message.content.clone())).await;
        Ok(response.message)
    })    
}
