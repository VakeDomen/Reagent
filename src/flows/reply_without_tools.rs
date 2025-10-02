use crate::{Agent, AgentError, InvocationBuilder, Message, NotificationHandler};

pub async fn reply_without_tools_flow(
    agent: &mut Agent,
    prompt: String,
) -> Result<Message, AgentError> {
    agent.history.push(Message::user(prompt));
    // let response = invoke_without_tools(agent).await?;
    let response = InvocationBuilder::default()
        .use_tools(false)
        .invoke_with(agent)
        .await?;

    agent
        .notify_done(true, response.message.content.clone())
        .await;
    Ok(response.message)
}
