use crate::{services::llm::message::Message, Agent, AgentError, Invocation, NotificationHandler};

pub async fn reply_without_tools_flow(
    agent: &mut Agent,
    prompt: String,
) -> Result<Message, AgentError> {
    agent.history.push(Message::user(prompt));
    let mut invocation = Invocation::chat().messages(agent.history.clone());
    if let Some(format) = agent.response_format.clone() {
        invocation = invocation.response_format(format);
    }
    let response = agent.invoke_model(invocation).await?;
    agent.history.push(response.message.clone());

    agent
        .notify_done(true, response.message.content.clone())
        .await;
    Ok(response.message)
}
