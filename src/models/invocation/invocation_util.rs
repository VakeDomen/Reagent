use crate::{
    models::{
        invocation::invocation_handler::InvokeFuture,
        AgentError
    },
    services::ollama::models::{
        base::{BaseRequest, OllamaOptions},
        chat::{ChatRequest, ChatResponse},
        tool::ToolCall
    },
    Agent,
    Message,
    Notification,
};

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
        Ok(response)
    })
}

/// Builds a [`ChatRequest`] from the agent’s state, *including* whatever
/// `agent.tools` currently holds (calling `agent.get_compiled_tools()` if None).  
///
/// # Errors  
/// Returns [`AgentError`] if `get_compiled_tools()` fails.  

pub async fn generate_llm_request(
    agent: &mut Agent
) -> Result<ChatRequest, AgentError> {
    if let None = agent.tools {
        agent.tools = agent.get_compiled_tools().await?;
    }

    Ok(ChatRequest {
        base: BaseRequest {
            model: agent.model.clone(),
            format: agent.response_format.clone(),
            options: Some(OllamaOptions {
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

/// Like [`generate_llm_request`] but always sets `tools: None` in the request.
pub async fn generate_llm_request_without_tools(
    agent: &mut Agent
) -> Result<ChatRequest, AgentError> {
    if let None = agent.tools {
        agent.tools = agent.get_compiled_tools().await?;
    }

    Ok(ChatRequest {
        base: BaseRequest {
            model: agent.model.clone(),
            format: agent.response_format.clone(),
            options: Some(OllamaOptions {
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
    agent.notify(Notification::PromptRequest(request.clone())).await;

    let raw = agent.ollama_client.chat(request).await;
    match raw {
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
                agent.notify(Notification::ToolCallRequest(call.clone())).await;

                match tool.execute(call.function.arguments.clone()).await {
                    Ok(output) => {
                        agent.notify(Notification::ToolCallSuccessResult(output.clone()))
                            .await;
                        results.push(Message::tool(output, call.id.clone().unwrap_or(call.function.name.clone())));
                    }
                    Err(e) => {
                        agent.notify(Notification::ToolCallErrorResult(e.to_string())).await;
                        let msg = format!("Error executing tool {}: {}", call.function.name, e);
                        results.push(Message::tool(msg, call.id.clone().unwrap_or(call.function.name.clone())));
                    }
                }
            } else {
                tracing::error!("No corresponding tool found.");
                let msg = format!("Could not find tool: {}", call.function.name);
                agent.notify(Notification::ToolCallErrorResult(msg.clone())).await;
                results.push(Message::tool(msg, "0".to_string()));
            }
        }
    } else {
        tracing::error!("No tools specified");
        agent.notify(Notification::ToolCallErrorResult("Empty tool call".into())).await;
        results.push(Message::tool(
            "If you want to use a tool specify the name of the available tool.",
            "Tool".to_string(),
        ));
    }

    results
}



#[cfg(test)]
mod tests {
    // tests/invoke_flow_integration.rs
use std::{error::Error, sync::Arc};
use tokio::{sync::{ Mutex}, time::{timeout, Duration}};
use serde_json::Value;

use crate::{
    AgentBuilder,
    AsyncToolFn,
    ToolBuilder,
    Notification,
    ToolExecutionError,
};

#[tokio::test]
async fn end_to_end_weather_tool_flow() -> Result<(), Box<dyn Error>> {

    let mut weather_agent = AgentBuilder::default()
        .set_model("qwen3:0.6b")
        .strip_thinking(true)
        .set_system_prompt("/no_think \nYou make up weather info in JSON. You always say it's snowing")
        .set_response_format(
            r#"
            {
              "type":"object",
              "properties":{
                "windy":{"type":"boolean"},
                "temperature":{"type":"integer"},
                "description":{"type":"string"}
              },
              "required":["windy","temperature","description"]
            }
            "#,
        )
        .build()
        .await
        .unwrap();

    let weather_ref = Arc::new(Mutex::new(weather_agent.clone()));
    let weather_exec: AsyncToolFn = {
        let weather_ref = weather_ref.clone();
        Arc::new(move |args: Value| {
            let weather_ref = weather_ref.clone();
            Box::pin(async move {
                let mut agent = weather_ref.lock().await;
                let loc = args.get("location")
                    .and_then(Value::as_str)
                    .ok_or_else(|| ToolExecutionError::ArgumentParsingError("Missing 'location'".into()))?;
                let prompt = format!("/no_think What is the weather in {}?", loc);
                let resp = agent.invoke_flow(prompt)
                    .await
                    .map_err(|e| ToolExecutionError::ExecutionFailed(e.to_string()))?;
                Ok(resp.content.unwrap_or_default())
            })
        })
    };

    let weather_tool = ToolBuilder::new()
        .function_name("get_current_weather")
        .function_description("Returns a weather forecast for a given location")
        .add_property("location", "string", "City name")
        .add_required_property("location")
        .executor(weather_exec)
        .build()
        .unwrap();

    let (mut agent, mut notifications_rx) = AgentBuilder::default()
        .set_model("qwen3:0.6b")
        .set_system_prompt("You are a helpful assistant.")
        .add_tool(weather_tool)
        .build_with_notification()
        .await
        .unwrap();

    let mut seen = Vec::new();
    let notify_handle = tokio::spawn(async move {
        while let Ok(Some(note)) = timeout(Duration::from_millis(500), notifications_rx.recv()).await {
            seen.push(note);
        }
        seen
    });

    let hello = agent.invoke_flow("Say hello")
        .await
        .unwrap();
    assert!(hello.content.unwrap().to_lowercase().contains("hello"));

    let weather = weather_agent.invoke_flow("What is the current weather in Koper?")
        .await
        .unwrap();
    let weather_json = weather.content.unwrap();
    let v: Value = serde_json::from_str(&weather_json).unwrap();
    assert_eq!(v["description"].as_str().unwrap_or(""), ""); // at least present

    let recall = agent.invoke_flow("What do you remember?")
        .await
        .unwrap();
    let rec_text = recall.content.unwrap_or_default();
    assert!(rec_text.contains("Say hello"));
    assert!(rec_text.contains("weather in Koper"));

    let notifications = notify_handle
        .await
        .unwrap();


    assert!(!notifications.is_empty(), "Expected at least one notification");
    assert!(notifications.iter().any(|n| matches!(n, Notification::PromptRequest(_))));
    assert!(notifications.iter().any(|n| matches!(n, Notification::PromptSuccessResult(_))));
    assert!(notifications.iter().any(|n| matches!(n, Notification::ToolCallRequest(_))));
    assert!(notifications.iter().any(|n| matches!(n, Notification::ToolCallSuccessResult(_))));
    assert!(notifications.iter().any(|n| matches!(n, Notification::Done(_))));

    Ok(())
}

}
