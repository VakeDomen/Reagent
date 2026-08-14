use crate::{services::llm::message::Message, Agent, AgentError, NotificationHandler};

pub async fn reply_without_tools_flow<I, O>(
    agent: &mut Agent<I, O>,
    prompt: String,
) -> Result<Message, AgentError> {
    let input = Message::user(prompt);
    agent.history.push(input);
    agent
        .model
        .set_history(agent.history.clone())
        .set_tools(None);
    let response = agent.model.invoke(()).await?;
    agent.history.push(response.message.clone());

    agent
        .notify_done(true, response.message.content.clone())
        .await;
    Ok(response.message)
}
