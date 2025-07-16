use crate::{
    models::{invocation::invocation_handler::InvokeFuture, AgentError}, 
    services::ollama::models::{
        base::{BaseRequest, OllamaOptions}, 
        chat::{ChatRequest, ChatResponse}, 
        tool::ToolCall
    }, 
    Agent,
    Message, 
    Notification
};


pub fn invoke<'a>(
    agent: &'a mut Agent,
) -> InvokeFuture<'a> {
    Box::pin(async move {
        let request = generate_llm_request(agent).await?;
        let response = call_model(agent, request).await?;
        Ok(response)
    })
}


pub fn invoke_without_tools<'a>(
    agent: &'a mut Agent,
) -> InvokeFuture<'a> {
    Box::pin(async move {
        let request = generate_llm_request_without_tools(agent).await?;
        let response = call_model(agent, request).await?;
        Ok(response)
    })
}

pub async fn generate_llm_request(agent: &mut Agent) -> Result<ChatRequest, AgentError> {
    if let None = agent.tools {
        agent.tools = agent.get_compiled_tools().await?;
    }
    Ok(ChatRequest {
        base: BaseRequest {
            model: agent.model.clone(),
            format: agent.response_format.clone(),
            options:  Some(OllamaOptions {
                num_ctx: agent.num_ctx,
                repeat_last_n: agent.repeat_last_n,
                repeat_penalty: agent.repeat_penalty,
                temperature: agent.temperature,
                seed: agent.seed,
                stop: agent.stop.clone(),
                num_predict: agent.num_predict,
                top_k: agent.top_k,
                top_p: agent.top_p,
                min_p: agent.min_p,
                presence_penalty: agent.presence_penalty,
                frequency_penalty: agent.frequency_penalty,
            }),
            stream: Some(false), 
            keep_alive: Some("5m".to_string()),
        },
        messages: agent.history.clone(),
        tools: agent.tools.clone(), 
    })
}

pub async fn generate_llm_request_without_tools(agent: &mut Agent) -> Result<ChatRequest, AgentError> {
    if let None = agent.tools {
        agent.tools = agent.get_compiled_tools().await?;
    }
    Ok(ChatRequest {
        base: BaseRequest {
            model: agent.model.clone(),
            format: agent.response_format.clone(),
            options:  Some(OllamaOptions {
                num_ctx: agent.num_ctx,
                repeat_last_n: agent.repeat_last_n,
                repeat_penalty: agent.repeat_penalty,
                temperature: agent.temperature,
                seed: agent.seed,
                stop: agent.stop.clone(),
                num_predict: agent.num_predict,
                top_k: agent.top_k,
                top_p: agent.top_p,
                min_p: agent.min_p,
                presence_penalty: agent.presence_penalty,
                frequency_penalty: agent.frequency_penalty,
            }),
            stream: Some(false), 
            keep_alive: Some("5m".to_string()),
        },
        messages: agent.history.clone(),
        tools: None, 
    })
}

pub async fn call_model(agent: &Agent, request: ChatRequest) -> Result<ChatResponse, AgentError> {
    agent.notify(Notification::PromptRequest(request.clone())).await;
    match agent.ollama_client.chat(request).await {
        Ok(mut resp) => {
            agent.notify(Notification::PromptSuccessResult(resp.clone())).await;
            
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
            agent.notify(Notification::PromptErrorResult(e.to_string())).await;
            Err(e.into())
        } 
    }
}

pub async fn call_tools(agent: &Agent, tool_calls: &Vec<ToolCall>) -> Vec<Message> {
    if let Some(avalible_tools) = &agent.tools {
        let mut messages = vec![];
        for tool_call in tool_calls {
            tracing::info!(
                target: "tool",                    
                tool = %tool_call.function.name,
                id   = ?tool_call.id,
                args = ?tool_call.function.arguments,
                "executing tool call"
            );
            let mut tool_found = false;
            for avalible_tool in avalible_tools {
                if !avalible_tool.function.name.eq(&tool_call.function.name) {
                    continue;
                }

                tool_found = true;
                agent.notify(Notification::ToolCallRequest(tool_call.clone())).await;

                match avalible_tool.execute(tool_call.function.arguments.clone()).await {
                    Ok(tool_result_content) => {
                        let response_tool_call_id = tool_call.id
                            .clone()
                            .unwrap_or_else(|| tool_call.function.name.clone());

                        agent.notify(Notification::ToolCallSuccessResult(tool_result_content.clone())).await;
                        messages.push(Message::tool(
                            tool_result_content,
                            response_tool_call_id, 
                        ));
                    }
                    Err(e) => {
                        tracing::error!(error = %e, "Tool execution failed");
                        let error_content = format!("Error executing tool {}: {}", tool_call.function.name, e);
                        let response_tool_call_id = tool_call.id.clone().unwrap_or_else(|| tool_call.function.name.clone());
                        
                        agent.notify(Notification::ToolCallErrorResult(e.to_string())).await;
                        messages.push(Message::tool(
                            error_content,
                            response_tool_call_id,
                        ));
                    }
                }
            }
            if !tool_found {
                tracing::error!("No corresponding tool found.");
                let message = format!("Could not find tool: {}", tool_call.function.name);
                agent.notify(Notification::ToolCallErrorResult(message.clone())).await;
                messages.push(Message::tool(
                    message, 
                    "0"
                ));
            }

        }
        messages
    } else {
        tracing::error!("No tools specified");
        agent.notify(Notification::ToolCallErrorResult("Empty tool call".to_string())).await;
        vec![Message::tool(
            "If you want to use a tool specifiy the name of the avalible tool.",
            "Tool",
        )]
    }
}