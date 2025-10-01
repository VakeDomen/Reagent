use futures::{pin_mut, StreamExt};

use crate::{
    notifications::Token,
    services::llm::{
        models::chat::{ChatRequest, ChatResponse, ChatStreamChunk},
        InferenceClientError,
    },
    Agent, AgentError, InvocationError, Message, NotificationHandler, ToolCall,
};

// /// Invoke the agent with its current configuration.
// ///
// /// Builds a [`ChatRequest`] from the agent state, sends it to the model,
// /// and appends the model’s response to the agent’s history.
// ///
// /// Depending on whether streaming is enabled (`agent.stream`),
// /// this will call the model in streaming or non-streaming mode.
// ///
// /// Returns a [`ChatResponse`] wrapped in an [`InvokeFuture`].
// pub(super) async fn invoke(agent: &mut Agent) -> Result<ChatResponse, AgentError> {
//     let request: ChatRequest = (&*agent).into();
//     let response = match &request.base.stream {
//         Some(true) => call_model_streaming(agent, request).await?,
//         _ => call_model_nonstreaming(agent, request).await?,
//     };
//     agent.history.push(response.message.clone());
//     Ok(response)
// }

// /// Invoke the agent and also execute any tool calls returned by the model.
// ///
// /// Like [`invoke`], but if the model response includes `tool_calls`,
// /// each one is executed via [`call_tools`], and the resulting tool
// /// messages are appended to the agent’s history.
// ///
// /// Returns the final [`ChatResponse`] (not including tool outputs).
// pub(super) async fn invoke_with_tool_calls(agent: &mut Agent) -> Result<ChatResponse, AgentError> {
//     let request: ChatRequest = (&*agent).into();
//     let response = match &request.base.stream {
//         Some(true) => call_model_streaming(agent, request).await?,
//         _ => call_model_nonstreaming(agent, request).await?,
//     };

//     agent.history.push(response.message.clone());

//     if let Some(tc) = response.message.tool_calls.clone() {
//         for tool_msg in call_tools(agent, &tc).await {
//             agent.history.push(tool_msg);
//         }
//     }

//     Ok(response)
// }

// /// Invoke the agent while disabling tool use.
// ///
// /// Builds a [`ChatRequest`] with tools cleared (`request.tools = None`)
// /// so the model cannot propose tool calls. The response is then appended
// /// to the agent’s history.
// ///
// /// Returns a [`ChatResponse`] wrapped in an [`InvokeFuture`].
// pub(super) async fn invoke_without_tools(agent: &mut Agent) -> Result<ChatResponse, AgentError> {
//     let mut request: ChatRequest = (&*agent).into();
//     request.tools = None;
//     let response = match &request.base.stream {
//         Some(true) => call_model_streaming(agent, request).await?,
//         _ => call_model_nonstreaming(agent, request).await?,
//     };
//     agent.history.push(response.message.clone());
//     Ok(response)
// }

pub(super) async fn call_model_nonstreaming(
    agent: &Agent,
    request: ChatRequest,
) -> Result<ChatResponse, InvocationError> {
    agent.notify_prompt_request(request.clone()).await;

    let raw = agent.model_client.chat(request).await;

    let mut resp = match raw {
        Ok(resp) => resp,
        Err(e) => {
            agent.notify_prompt_error(e.to_string()).await;
            return Err(e.into());
        }
    };

    agent.notify_poompt_success(resp.clone()).await;

    if agent.strip_thinking {
        if let Some(content) = resp.message.content.clone() {
            if let Some(after) = content.split("</think>").nth(1) {
                resp.message.content = Some(after.to_string());
            }
        }
    }

    Ok(resp)
}

pub(super) async fn call_model_streaming(
    agent: &Agent,
    request: ChatRequest,
) -> Result<ChatResponse, InvocationError> {
    agent.notify_prompt_request(request.clone()).await;

    let stream = match agent.model_client.chat_stream(request).await {
        Ok(s) => s,
        Err(e) => {
            agent.notify_prompt_error(e.to_string()).await;
            return Err(e.into());
        }
    };

    pin_mut!(stream);

    let mut full_content = None;
    let mut latest_message: Option<Message> = None;
    let mut tool_calls: Option<Vec<ToolCall>> = None;
    let mut done_chunk: Option<ChatStreamChunk> = None;

    while let Some(chunk_res) = stream.next().await {
        let chunk = match chunk_res {
            Ok(c) => c,
            Err(e) => {
                agent.notify_prompt_error(e.to_string()).await;
                return Err(e.into());
            }
        };

        if chunk.done {
            done_chunk = Some(chunk);
            break;
        }

        if let Some(msg) = &chunk.message {
            if let Some(calls) = &msg.tool_calls {
                match tool_calls.as_mut() {
                    Some(tool_call_vec) => tool_call_vec.extend(calls.clone()),
                    None => tool_calls = Some(calls.clone()),
                }
            }

            if let Some(tok) = &msg.content {
                agent
                    .notify_token(Token {
                        tag: None,
                        value: tok.clone(),
                    })
                    .await;
                match full_content.as_mut() {
                    None => full_content = Some(tok.to_owned()),
                    Some(content) => content.push_str(tok),
                }
            }

            latest_message = Some(msg.clone());
        }
    }

    let Some(chunk) = done_chunk else {
        return Err(
            InferenceClientError::Api("stream ended without a final `done` chunk".into()).into(),
        );
    };

    let mut final_msg = latest_message.unwrap_or_else(|| Message::assistant(String::new()));
    final_msg.content = full_content;
    final_msg.tool_calls = tool_calls;

    if agent.strip_thinking {
        if let Some(c) = &final_msg.content {
            if let Some(after) = c.split("</think>").nth(1) {
                final_msg.content = Some(after.to_string());
            }
        }
    }

    let mut response = ChatResponse {
        model: chunk.model,
        created_at: chunk.created_at,
        message: final_msg,
        done: chunk.done,
        done_reason: chunk.done_reason,
        total_duration: chunk.total_duration,
        load_duration: chunk.load_duration,
        prompt_eval_count: chunk.prompt_eval_count,
        prompt_eval_duration: chunk.prompt_eval_duration,
        eval_count: chunk.eval_count,
        eval_duration: chunk.eval_duration,
    };

    agent.notify_poompt_success(response.clone()).await;

    if agent.strip_thinking {
        if let Some(content) = response.message.content.clone() {
            if let Some(after) = content.split("</think>").nth(1) {
                response.message.content = Some(after.to_string());
            }
        }
    }

    Ok(response)
}
