use futures::{pin_mut, StreamExt};

use crate::{
    models::{agents::flow::invocation_flows::InvokeFuture, notification::Token, AgentError}, 
    services::ollama::models::{
        chat::{ChatRequest, ChatResponse, ChatStreamChunk}, 
        errors::OllamaError,
        tool::ToolCall
    }, 
    Agent, 
    Message, 
    NotificationContent
};
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
        let request = generate_llm_request(agent).await;
        let response = match &request.base.stream {
            Some(true) => call_model_streaming(agent, request).await?,
            _ => call_model(agent, request).await?,
        };
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
        let request = generate_llm_request(agent).await;
        let response = match &request.base.stream {
            Some(true) => call_model_streaming(agent, request).await?,
            _ => call_model(agent, request).await?,
        };
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
        let request = generate_llm_request_without_tools(agent).await;
        let response = match &request.base.stream {
            Some(true) => call_model_streaming(agent, request).await?,
            _ => call_model(agent, request).await?,
        };
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

/// Streams `/api/chat`, emits NotificationToken::Token for every chunk,
/// then returns the reconstructed ChatResponse (so callers behave exactly
/// like with the non-streaming call).
///
/// * Sends the same PromptRequest / PromptSuccessResult / PromptErrorResult
///   notifications as `call_model`.
/// * Honors `agent.strip_thinking` on the final text.
pub async fn call_model_streaming(
    agent: &Agent,
    mut request: ChatRequest,
) -> Result<ChatResponse, AgentError> {
    request.base.stream = Some(true);

    agent
        .notify(NotificationContent::PromptRequest(request.clone()))
        .await;

    let stream = match agent.ollama_client.chat_stream(request).await {
        Ok(s)  => s,
        Err(e) => {
            agent.notify(NotificationContent::PromptErrorResult(e.to_string())).await;
            return Err(e.into());
        }
    };


    pin_mut!(stream);  

    let mut full_content = String::new();
    let mut latest_message: Option<Message> = None;

    let mut last_chunk: Option<ChatStreamChunk> = None;

    while let Some(chunk_res) = stream.next().await {
        match chunk_res {
            Ok(chunk) => {
                if let Some(msg) = chunk.message.clone() {
                    // Token-level work
                    if let Some(tok) = &msg.content {
                        agent.notify(NotificationContent::Token(Token {tag: None, value: tok.clone()})).await;
                        full_content.push_str(tok);
                    }

                    latest_message = Some(msg);
                }

                if chunk.done { 
                    last_chunk = Some(chunk); 
                    break; 
                }
            }
            Err(e) => {
                agent
                    .notify(NotificationContent::PromptErrorResult(e.to_string()))
                    .await;
                return Err(e.into());
            }
        }
    }

    let Some(chunk) = last_chunk else {
        return Err(OllamaError::Api("stream ended without a final `done` chunk".into()).into());
    };

    let mut final_msg = latest_message.unwrap_or_else(|| Message::assistant(String::new()));

    // glue together the accumulated text + any trailing content
    let trailing = final_msg.content.unwrap_or_default();
    final_msg.content = Some(format!("{full_content}{trailing}"));

    if agent.strip_thinking {
        if let Some(c) = &final_msg.content {
            if let Some(after) = c.split("</think>").nth(1) {
                final_msg.content = Some(after.to_string());
            }
        }
    }


    let mut response = ChatResponse {
        model:         chunk.model,
        created_at:    chunk.created_at,
        message:       Message::assistant(full_content),
        done:          true,
        done_reason:   chunk.done_reason,
        total_duration:    chunk.total_duration,
        load_duration:     chunk.load_duration,
        prompt_eval_count: chunk.prompt_eval_count,
        prompt_eval_duration: chunk.prompt_eval_duration,
        eval_count:        chunk.eval_count,
        eval_duration:     chunk.eval_duration,
    };

    if agent.strip_thinking {
        if let Some(content) = response.message.content.clone() {
            if let Some(after) = content.split("</think>").nth(1) {
                response.message.content = Some(after.to_string());
            }
        }
    }

    agent.notify(NotificationContent::PromptSuccessResult(response.clone())).await;

    Ok(response)
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
