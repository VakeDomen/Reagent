use crate::{
    call_tools, services::llm::message::Message, Agent, AgentError, InvocationBuilder,
    NotificationHandler,
};

pub async fn default_flow(agent: &mut Agent, prompt: String) -> Result<Message, AgentError> {
    agent.history.push(Message::user(prompt));
    let mut response = InvocationBuilder::default().invoke_with(agent).await?;
    if let Some(tc) = response.message.tool_calls {
        for tool_msg in call_tools(agent, &tc).await {
            agent.history.push(tool_msg);
        }
        response = InvocationBuilder::default()
            .use_tools(false)
            .invoke_with(agent)
            .await?;
    }

    agent
        .notify_done(true, response.message.content.clone())
        .await;
    Ok(response.message)
}
