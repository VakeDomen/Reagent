use crate::{models::agents::flow::invocation_flows::FlowFuture, util::invocations::{call_tools, invoke}, Agent, Message};


pub fn default_flow<'a>(agent: &'a mut Agent, prompt: String) -> FlowFuture<'a> {
    Box::pin(async move {
        agent.history.push(Message::user(prompt));
        let mut response = invoke(agent).await?;
        if let Some(tc) = response.message.tool_calls {
            for tool_msg in call_tools(agent, &tc).await {
                agent.history.push(tool_msg);
            }
            response = invoke(agent).await?;
        } 

        agent.notify(crate::NotificationContent::Done(true, response.message.content.clone())).await;
        Ok(response.message)
    })    
}