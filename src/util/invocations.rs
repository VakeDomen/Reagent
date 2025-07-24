use crate::{models::{agents::flow::invocation_flows::InvokeFuture, AgentError}, services::ollama::models::{chat::{ChatRequest, ChatResponse}, tool::ToolCall}, Agent, Message, Notification, NotificationContent};
use crate::util::request_generation::{generate_llm_request, generate_llm_request_without_tools};


/// Invoke the agent’s normal LLM flow:  
/// 1. build a request via [`generate_llm_request`],  
/// 2. call the model via [`call_model`].  
///  
/// # Returns  
/// A pinned, boxed future that resolves to a [`ChatResponse`].  
pub fn invoke<'a>(
    agent: &'a mut Agent,
) -> InvokeFuture<'a> {
    Box::pin(async move {
        let request = generate_llm_request(agent).await?;
        let response = call_model(agent, request).await?;
        agent.history.push(response.message.clone());
        Ok(response)
    })
}


/// Invoke the agent’s normal LLM flow and call tools the LLM requested
/// 1. build a request via [`generate_llm_request`],  
/// 2. call the model via [`call_model`].  
/// 3. optionally call the tools [`call_tools`]
///  
/// # Returns  
/// A pinned, boxed future that resolves to a [`ChatResponse`].  
pub fn invoke_with_tool_calls<'a>(
    agent: &'a mut Agent,
) -> InvokeFuture<'a> {
    Box::pin(async move {
        let request = generate_llm_request(agent).await?;
        let response = call_model(agent, request).await?;

        agent.history.push(response.message.clone());

        if let Some(tc) = response.message.tool_calls.clone() {
            for tool_msg in call_tools(agent, &tc).await {
                agent.history.push(tool_msg);
            }
        } 
        
        Ok(response)
    })
}



/// Exactly like [`invoke`], but omits all tool definitions in the request.
/// Useful when you know no tools should be present.  
pub fn invoke_without_tools<'a>(
    agent: &'a mut Agent,
) -> InvokeFuture<'a> {
    Box::pin(async move {
        let request = generate_llm_request_without_tools(agent).await?;
        let response = call_model(agent, request).await?;
        agent.history.push(response.message.clone());
        Ok(response)
    })
}



/// Actually dispatches to `agent.ollama_client.chat(...)`, emits
/// `Notification::PromptRequest`, then on success `PromptSuccessResult`,
/// on failure `PromptErrorResult`. Strips any `<think>…</think>` prefix
/// if `agent.strip_thinking` is set.
///
/// # Errors  
/// Bubbles up any error from the underlying client as [`AgentError`].
pub async fn call_model(
    agent: &Agent,
    request: ChatRequest,
) -> Result<ChatResponse, AgentError> {
    agent.notify(NotificationContent::PromptRequest(request.clone())).await;

    let raw = agent.ollama_client.chat(request).await;
    match raw {
        Ok(mut resp) => {
            agent.notify(NotificationContent::PromptSuccessResult(resp.clone())).await;

            if agent.strip_thinking {
                if let Some(content) = resp.message.content.clone() {
                    if let Some(after) = content.split("</think>").nth(1) {
                        resp.message.content = Some(after.to_string());
                    }
                }
            }

            Ok(resp)
        }
        Err(e) => {
            agent.notify(NotificationContent::PromptErrorResult(e.to_string())).await;
            Err(e.into())
        }
    }
}

/// For each `ToolCall` in `tool_calls`, attempts to find a matching
/// `Tool` in `agent.tools`.  If found, invokes `tool.execute(...)`:
/// on `Ok(v)`, emits `ToolCallSuccessResult` & returns a tool‐message;
/// on `Err(e)`, emits `ToolCallErrorResult` & returns an error‐message.
/// If no matching tool is found at all, returns a single `.tool(...)`
/// message complaining that the tool is missing.
///
/// # Panics  
/// Never panics; always returns at least one `Message::tool(...)`.
pub async fn call_tools(
    agent: &Agent,
    tool_calls: &[ToolCall]
) -> Vec<Message> {
    let mut results = Vec::new();

    if let Some(avail) = &agent.tools {
        for call in tool_calls {
            tracing::info!(
                target: "tool",
                tool = %call.function.name,
                id   = ?call.id,
                args = ?call.function.arguments,
                "executing tool call",
            );

            // try to find the tool
            if let Some(tool) = avail.iter().find(|t| t.function.name == call.function.name) {
                agent.notify(NotificationContent::ToolCallRequest(call.clone())).await;

                match tool.execute(call.function.arguments.clone()).await {
                    Ok(output) => {
                        agent.notify(NotificationContent::ToolCallSuccessResult(output.clone()))
                            .await;
                        results.push(Message::tool(output, call.id.clone().unwrap_or(call.function.name.clone())));
                    }
                    Err(e) => {
                        agent.notify(NotificationContent::ToolCallErrorResult(e.to_string())).await;
                        let msg = format!("Error executing tool {}: {}", call.function.name, e);
                        results.push(Message::tool(msg, call.id.clone().unwrap_or(call.function.name.clone())));
                    }
                }
            } else {
                tracing::error!("No corresponding tool found.");
                let msg = format!("Could not find tool: {}", call.function.name);
                agent.notify(NotificationContent::ToolCallErrorResult(msg.clone())).await;
                results.push(Message::tool(msg, "0".to_string()));
            }
        }
    } else {
        tracing::error!("No tools specified");
        agent.notify(NotificationContent::ToolCallErrorResult("Empty tool call".into())).await;
        results.push(Message::tool(
            "If you want to use a tool specify the name of the available tool.",
            "Tool".to_string(),
        ));
    }

    results
}
