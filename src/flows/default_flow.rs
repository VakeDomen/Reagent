use crate::{
    services::llm::message::Message, Agent, AgentError, InvocationBuilder, NotificationHandler,
    Role,
};

pub async fn default_flow(agent: &mut Agent, prompt: String) -> Result<Message, AgentError> {
    agent.history.push(Message::user(prompt));
    let mut response = InvocationBuilder::default().invoke_with(agent).await?;

    if let Some(last_message) = agent.history.last() {
        if last_message.role == Role::Tool {
            response = InvocationBuilder::default()
                .use_tools(false)
                .invoke_with(agent)
                .await?;
        }
    }

    agent
        .notify_done(true, response.message.content.clone())
        .await;
    Ok(response.message)
}
