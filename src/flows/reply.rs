use crate::{invoke, Agent, AgentError, Message};

pub async fn reply_flow(agent: &mut Agent, prompt: String) -> Result<Message, AgentError> {
    agent.history.push(Message::user(prompt));
    let response = invoke(agent).await?;
    agent.notify(crate::NotificationContent::Done(true, response.message.content.clone())).await;
    Ok(response.message)    
}
