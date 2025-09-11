use crate::{call_tools, invoke, invoke_without_tools, Agent, AgentError, Message};

pub async fn default_flow(agent: &mut Agent, prompt: String) -> Result<Message, AgentError> {
    agent.history.push(Message::user(prompt));
    let mut response = invoke(agent).await?;
    if let Some(tc) = response.message.tool_calls {
        for tool_msg in call_tools(agent, &tc).await {
            agent.history.push(tool_msg);
        }
        response = invoke_without_tools(agent).await?;
    } 

    agent.notify(crate::NotificationContent::Done(true, response.message.content.clone())).await;
    Ok(response.message)
}