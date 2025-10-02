use crate::{Agent, AgentError, InvocationBuilder, Message, NotificationHandler};

pub async fn reply_flow(agent: &mut Agent, prompt: String) -> Result<Message, AgentError> {
    agent.history.push(Message::user(prompt));
    let response = InvocationBuilder::default().invoke_with(agent).await?;
    agent
        .notify_done(true, response.message.content.clone())
        .await;
    Ok(response.message)
}
