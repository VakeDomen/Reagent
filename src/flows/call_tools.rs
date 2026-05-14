use crate::{
    call_tools, services::llm::message::Message, Agent, AgentError, InvocationBuilder,
    NotificationHandler,
};

pub async fn call_tools_flow(agent: &mut Agent, prompt: String) -> Result<Message, AgentError> {
    agent.history.push(Message::user(prompt));
    let response = InvocationBuilder::default().invoke_with(agent).await?;
    if let Some(tool_calls) = response
        .message
        .tool_calls
        .as_ref()
        .filter(|calls| !calls.is_empty())
    {
        for tool_msg in call_tools(agent, tool_calls).await {
            agent.history.push(tool_msg);
        }
    }

    agent
        .notify_done(true, response.message.content.clone())
        .await;
    Ok(response.message)
}
