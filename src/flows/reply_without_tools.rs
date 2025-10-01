use crate::{invoke_without_tools, Agent, AgentError, Message, NotificationHandler};

pub async fn reply_without_tools_flow(
    agent: &mut Agent,
    prompt: String,
) -> Result<Message, AgentError> {
    agent.history.push(Message::user(prompt));
    let response = invoke_without_tools(agent).await?;
    agent
        .notify_done(true, response.message.content.clone())
        .await;
    Ok(response.message)
}
