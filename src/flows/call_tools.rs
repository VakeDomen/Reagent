use crate::{
    services::llm::message::Message, Agent, AgentError, InvocationBuilder, NotificationHandler,
};

pub async fn call_tools_flow(agent: &mut Agent, prompt: String) -> Result<Message, AgentError> {
    agent.history.push(Message::user(prompt));
    let response = InvocationBuilder::default().invoke_with(agent).await?;

    agent
        .notify_done(true, response.message.content.clone())
        .await;
    Ok(response.message)
}
