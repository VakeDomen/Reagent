use crate::{
    call_tools, services::llm::message::Message, Agent, AgentError, Invocation, NotificationHandler,
};

pub async fn call_tools_flow(agent: &mut Agent, prompt: String) -> Result<Message, AgentError> {
    agent.history.push(Message::user(prompt));
    let mut invocation = Invocation::chat().messages(agent.history.clone());
    if let Some(tools) = agent.tools.clone().filter(|tools| !tools.is_empty()) {
        invocation = invocation.tools(tools);
    }
    if let Some(format) = agent.response_format.clone() {
        invocation = invocation.response_format(format);
    }
    let response = agent.invoke_model(invocation).await?;
    agent.history.push(response.message.clone());
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
