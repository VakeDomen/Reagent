use futures::{pin_mut, StreamExt};

use crate::{
    notifications::Token,
    services::llm::{
        models::chat::{ChatResponse, ChatStreamChunk},
        InferenceClientError,
    },
    InvocationError, InvocationRequest, Message, NotificationHandler, ToolCall,
};

pub(super) async fn invoke_nonstreaming(
    invocation_request: InvocationRequest,
) -> Result<ChatResponse, InvocationError> {
    let InvocationRequest {
        strip_thinking,
        request,
        client,
        notification_channel,
    } = invocation_request;

    notification_channel
        .notify_prompt_request(request.clone())
        .await;

    let raw = client.chat(request).await;

    let mut resp = match raw {
        Ok(resp) => resp,
        Err(e) => {
            notification_channel
                .notify_prompt_error(e.to_string())
                .await;
            return Err(e.into());
        }
    };

    notification_channel
        .notify_poompt_success(resp.clone())
        .await;

    if strip_thinking {
        if let Some(content) = resp.message.content.clone() {
            if let Some(after) = content.split("</think>").nth(1) {
                resp.message.content = Some(after.to_string());
            }
        }
    }

    Ok(resp)
}

pub(super) async fn invoke_streaming(
    invocation_request: InvocationRequest,
) -> Result<ChatResponse, InvocationError> {
    let InvocationRequest {
        strip_thinking,
        request,
        client,
        notification_channel,
    } = invocation_request;

    notification_channel
        .notify_prompt_request(request.clone())
        .await;

    let stream = match client.chat_stream(request).await {
        Ok(s) => s,
        Err(e) => {
            notification_channel
                .notify_prompt_error(e.to_string())
                .await;
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
                notification_channel
                    .notify_prompt_error(e.to_string())
                    .await;
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
                notification_channel
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

    if strip_thinking {
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

    notification_channel
        .notify_poompt_success(response.clone())
        .await;

    if strip_thinking {
        if let Some(content) = response.message.content.clone() {
            if let Some(after) = content.split("</think>").nth(1) {
                response.message.content = Some(after.to_string());
            }
        }
    }

    Ok(response)
}
