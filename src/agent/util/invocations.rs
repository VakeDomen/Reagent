use futures::{pin_mut, StreamExt};
use serde::Serialize;
use tracing::{error, span, Level, Span};
use tracing_opentelemetry::OpenTelemetrySpanExt;

use crate::{
    notifications::Token,
    services::llm::{
        message::Message,
        models::chat::{ChatResponse, ChatStreamChunk},
        InferenceClientError,
    },
    ChatRequest, InvocationError, InvocationRequest, NotificationHandler, ToolCall,
};

#[derive(Debug, Serialize)]
struct UsageDetails {
    prompt_tokens: i64,
    completion_tokens: i64,
    total_tokens: i64,
}

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

    let gen_span = set_telemetry_request_attributes(&request);
    let _guard = gen_span.enter();

    let raw = client.chat(request).await;

    let mut resp = match raw {
        Ok(resp) => resp,
        Err(e) => {
            notification_channel
                .notify_prompt_error(e.to_string())
                .await;
            extract_error_telemetry(&gen_span, e.to_string().as_str());
            return Err(e.into());
        }
    };

    extract_response_telemetry(&gen_span, &resp);

    notification_channel
        .notify_prompt_success(resp.clone())
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

    let gen_span = set_telemetry_request_attributes(&request);
    let _guard = gen_span.enter();

    let stream = match client.chat_stream(request).await {
        Ok(s) => s,
        Err(e) => {
            notification_channel
                .notify_prompt_error(e.to_string())
                .await;
            extract_error_telemetry(&gen_span, e.to_string().as_str());
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
        let error_message = "stream ended without a final `done` chunk";
        extract_error_telemetry(&gen_span, error_message);
        return Err(InferenceClientError::Api(error_message.into()).into());
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

    extract_response_telemetry(&gen_span, &response);

    notification_channel
        .notify_prompt_success(response.clone())
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

fn extract_error_telemetry(gen_span: &Span, error_message: &str) {
    gen_span.set_attribute("otel.status_code", "ERROR");
    gen_span.set_attribute("error.message", error_message.to_string());
    gen_span.set_status(opentelemetry::trace::Status::Error {
        description: error_message.to_string().into(),
    });
    error!("stream ended without a final `done` chunk");
}

fn extract_response_telemetry(gen_span: &Span, response: &ChatResponse) {
    let prompt_tokens = response.prompt_eval_count.unwrap_or(0);
    let completion_tokens = response.eval_count.unwrap_or(0);
    let total_tokens = prompt_tokens + completion_tokens;

    let usage = UsageDetails {
        prompt_tokens: prompt_tokens as i64,
        completion_tokens: completion_tokens as i64,
        total_tokens: total_tokens as i64,
    };

    gen_span.set_attribute(
        "langfuse.observation.output",
        serde_json::to_string_pretty(&response.message.content)
            .unwrap_or(format!("{:#?}", response)),
    );
    gen_span.set_attribute("gen_ai.completion.0.role", "assistant");
    gen_span.set_attribute(
        "gen_ai.completion.0.content",
        response.message.content.clone().unwrap_or_default(),
    );

    gen_span.set_attribute(
        "langfuse.observation.usage_details",
        serde_json::to_string(&usage).unwrap_or(format!("{:#?}", usage)),
    );

    if let Some(duration) = response.total_duration {
        gen_span.set_attribute("llm.duration.total_s", (duration as f64) / 1_000_000_000.0);
    }
    if let Some(duration) = response.load_duration {
        gen_span.set_attribute("llm.duration.load_s", (duration as f64) / 1_000_000_000.0);
    }
    if let Some(duration) = response.prompt_eval_duration {
        gen_span.set_attribute(
            "llm.duration.prompt_eval_ms",
            (duration as f64) / 1_000_000.0,
        );
    }
    if let Some(duration) = response.eval_duration {
        gen_span.set_attribute("llm.duration.eval_ms", (duration as f64) / 1_000_000.0);
    }

    gen_span.set_attribute("otel.status_code", "OK");
}

fn set_telemetry_request_attributes(request: &ChatRequest) -> Span {
    let gen_span = span!(
        Level::INFO,
        "Chat Request",
        "langfuse.observation.type" = "generation",
        "langfuse.observation.model.name" = request.base.model.as_str(),
        "langfuse.observation.input" =
            serde_json::to_string(&request.messages).unwrap_or(format!("{:#?}", request.messages)),
        "langfuse.observation.model.parameters" = serde_json::to_string(&request.base.options)
            .unwrap_or(format!("{:#?}", request.base.options)),
    );

    gen_span
}
