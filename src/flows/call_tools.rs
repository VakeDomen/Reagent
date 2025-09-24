use crate::{call_tools, invoke, Agent, AgentError, Message, NotificationHandler};


pub async fn call_tools_flow(agent: &mut Agent, prompt: String) -> Result<Message, AgentError> {
    agent.history.push(Message::user(prompt));
    let response = invoke(agent).await?;
    if let Some(tc) = response.message.tool_calls.clone() {
        for tool_msg in call_tools(agent, &tc).await {
            agent.history.push(tool_msg);
        }
    } 
    agent
        .notify_done(
            true, 
            response.message.content.clone()
        )
        .await;
    Ok(response.message)
}