use crate::{
    call_tools, services::llm::message::Message, Agent, AgentError, InvocationBuilder,
    NotificationHandler, ToolCall,
};

const DEFAULT_MAX_ITERATIONS: usize = 50;

pub async fn default_flow(agent: &mut Agent, prompt: String) -> Result<Message, AgentError> {
    agent.history.push(Message::user(prompt));
    let max_iterations = agent
        .max_iterations
        .unwrap_or(DEFAULT_MAX_ITERATIONS)
        .max(1);
    let mut response = None;

    for iteration in 0..max_iterations {
        let allow_tools = iteration + 1 < max_iterations;
        let current = InvocationBuilder::default()
            .use_tools(allow_tools)
            .invoke_with(agent)
            .await?;
        let tool_calls = executable_tool_calls(&current.message, allow_tools);
        response = Some(current);

        let Some(tool_calls) = tool_calls else {
            break;
        };

        for tool_msg in call_tools(agent, &tool_calls).await {
            agent.history.push(tool_msg);
        }
    }

    let message = response
        .expect("default flow always performs at least one iteration")
        .message;

    agent.notify_done(true, message.content.clone()).await;
    Ok(message)
}

fn executable_tool_calls(message: &Message, allow_tools: bool) -> Option<Vec<ToolCall>> {
    allow_tools
        .then(|| message.tool_calls.as_ref())
        .flatten()
        .filter(|calls| !calls.is_empty())
        .cloned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ToolCall, ToolCallFunction, ToolType};

    #[test]
    fn empty_tool_calls_do_not_request_tools() {
        let mut message = Message::assistant("done");
        message.tool_calls = Some(Vec::new());

        assert!(executable_tool_calls(&message, true).is_none());
    }

    #[test]
    fn non_empty_tool_calls_request_tools() {
        let mut message = Message::assistant("calling");
        message.tool_calls = Some(vec![ToolCall {
            id: Some("call_1".into()),
            tool_type: ToolType::Function,
            function: ToolCallFunction {
                name: "bash".into(),
                arguments: serde_json::json!({ "command": "pwd" }),
            },
        }]);

        assert!(executable_tool_calls(&message, true).is_some());
    }

    #[test]
    fn tool_calls_are_not_executable_when_tools_are_disabled() {
        let mut message = Message::assistant("calling");
        message.tool_calls = Some(vec![ToolCall {
            id: Some("call_1".into()),
            tool_type: ToolType::Function,
            function: ToolCallFunction {
                name: "bash".into(),
                arguments: serde_json::json!({ "command": "pwd" }),
            },
        }]);

        assert!(executable_tool_calls(&message, false).is_none());
    }
}
