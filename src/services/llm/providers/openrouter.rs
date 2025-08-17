use std::pin::Pin;
use futures::{Stream, StreamExt};
use reqwest::{header::{HeaderMap, HeaderName, HeaderValue, AUTHORIZATION, CONTENT_TYPE}, Client};
use serde::{Deserialize, Serialize};
use tracing::{debug, instrument};

use crate::services::llm::client::{ClientConfig};
use crate::services::llm::models::base::{InferenceOptions, Message, Role};
use crate::services::llm::models::chat::{ChatRequest, ChatResponse, ChatStreamChunk};
use crate::services::llm::models::embedding::{EmbeddingsRequest, EmbeddingsResponse};
use crate::services::llm::models::errors::ModelClientError;

#[derive(Debug, Clone)]
pub struct OpenRouterClient {
    client: Client,
    base_url: String,
}

impl OpenRouterClient {
    pub fn new(cfg: ClientConfig) -> Result<Self, ModelClientError> {
        let api_key = cfg
            .api_key
            .ok_or_else(|| ModelClientError::Config("OpenRouter requires api_key".into()))?;
        let base_url = cfg
            .base_url
            .unwrap_or_else(|| "https://openrouter.ai/api/v1".to_string());

        let mut headers = HeaderMap::new();
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {api_key}"))
                .map_err(|e| ModelClientError::Config(format!("Invalid api_key header: {e}")))?,
        );
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        if let Some(extra) = cfg.extra_headers {
            for (k, v) in extra.into_iter() {
                let name = HeaderName::from_bytes(k.as_bytes())
                    .map_err(|_| ModelClientError::Config(format!("Invalid header name: {k}")))?;
                let value = HeaderValue::from_str(&v)
                    .map_err(|_| ModelClientError::Config(format!("Invalid header value for {k}")))?;
                headers.insert(name, value);
            }
        }

        let client = Client::builder()
            .default_headers(headers)
            .build()?;

        Ok(Self { client, base_url })
    }

    fn map_messages(msgs: &[Message]) -> Vec<OrMessage> {
        msgs.iter()
            .map(|m| OrMessage {
                role: match m.role {
                    Role::System => "system".to_string(),
                    Role::Developer => "system".to_string(),
                    Role::User => "user".to_string(),
                    Role::Assistant => "assistant".to_string(),
                    Role::Tool => "tool".to_string(),
                },
                content: m.content.clone().unwrap_or_default(),
            })
            .collect()
    }

    fn map_options(opts: &Option<InferenceOptions>) -> OrParams {
        let mut p = OrParams::default();
        if let Some(o) = opts {
            p.temperature = o.temperature;
            p.top_p = o.top_p;
            p.top_k = o.top_k;
            p.max_tokens = o.max_tokens.or(o.num_predict);
            p.frequency_penalty = o.frequency_penalty;
            p.presence_penalty = o.presence_penalty;
            p.stop = o.stop.clone();
            // num_ctx, min_p, repeat penalties are not standard in OR OpenAI schema
        }
        p
    }

    #[instrument(name = "openrouter.chat", skip_all)]
    async fn chat_inner(&self, req: ChatRequest, stream: bool) -> Result<reqwest::Response, ModelClientError> {
        let url = format!(
            "{}/chat/completions",
            self.base_url.trim_end_matches('/')
        );
        let mut body = OrChatRequest::from(req);
        body.stream = Some(stream);
        let resp = self.client.post(url).json(&body).send().await?;
        Ok(resp)
    }

    
    pub async fn chat(&self, req: ChatRequest) -> Result<ChatResponse, ModelClientError> {
        let resp = self.chat_inner(req, false).await?;
        let status = resp.status();
        let text = resp.text().await?;

        // HTTP error
        if !status.is_success() {
            if let Some(e) = parse_oopen_router_error(&text) {
                return Err(e);
            }
            return Err(ModelClientError::Api(format!("Request failed: {status} - {text}")));
        }

        // HTTP 200 but body is an error envelope
        if let Some(e) = parse_oopen_router_error(&text) {
            return Err(e);
        }

        let or: OrChatResponse = serde_json::from_str(&text)
            .map_err(|e| ModelClientError::Serialization(format!("decode error: {e}; raw: {text}")))?;

        let message = or.choices.first()
            .map(|c| Message {
                role: Role::Assistant,
                content: Some(c.message.content.clone()),
                images: None,
                tool_calls: None,
                tool_call_id: None,
            })
            .unwrap_or(Message::assistant(String::new()));

        Ok(ChatResponse {
            model: or.model,
            created_at: or.created.to_string(),
            message,
            done: true,
            done_reason: or.choices.first().and_then(|c| c.finish_reason.clone()),
            total_duration: None,
            load_duration: None,
            prompt_eval_count: None,
            prompt_eval_duration: None,
            eval_count: None,
            eval_duration: None,
        })
    }


    pub async fn chat_stream(
        &self,
        req: ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatStreamChunk, ModelClientError>> + Send + 'static>>, ModelClientError> {
        use async_stream::try_stream;
        let resp = self.chat_inner(req, true).await?;
        let status = resp.status();

        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            if let Some(e) = parse_oopen_router_error(&text) {
                return Err(e);
            }
            return Err(ModelClientError::Api(format!("Request failed: {status} - {text}")));
        }

        let byte_stream = resp.bytes_stream();
        let s = try_stream! {
            let mut buf = Vec::<u8>::new();
            futures::pin_mut!(byte_stream);

            while let Some(chunk) = byte_stream.next().await {
                let chunk = chunk.map_err(|e| ModelClientError::Request(e.to_string()))?;
                buf.extend_from_slice(&chunk);

                while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
                    let line: Vec<u8> = buf.drain(..=pos).collect();
                    let line = String::from_utf8_lossy(&line).trim().to_string();

                    if line.is_empty() || line.starts_with(':') { continue; }
                    if !line.starts_with("data:") { continue; }

                    let data = line[5..].trim();

                    if data.contains("[DONE]") {
                        yield ChatStreamChunk {
                            model: String::new(),
                            created_at: String::new(),
                            message: None,
                            done: true,
                            done_reason: Some("stop".into()),
                            total_duration: None,
                            load_duration: None,
                            prompt_eval_count: None,
                            prompt_eval_duration: None,
                            eval_count: None,
                            eval_duration: None,
                        };
                        return;
                    }

                    // if the SSE data payload is an error envelope, surface it and stop
                    if let Some(e) = parse_oopen_router_error(data) {
                        Err(e)?;
                    }

                    let parsed: OrStreamChunk = match serde_json::from_str(data) {
                        Ok(v) => v,
                        Err(e) => { debug!(err = %e, raw = %data, "stream json decode error"); continue; }
                    };

                    let mut out_msg: Option<Message> = None;
                    if let Some(choice) = parsed.choices.first() {
                        if let Some(content) = choice.delta.content.clone() {
                            out_msg = Some(Message::assistant(content));
                        }
                    }

                    yield ChatStreamChunk {
                        model: parsed.model,
                        created_at: parsed.created.to_string(),
                        message: out_msg,
                        done: false,
                        done_reason: None,
                        total_duration: None,
                        load_duration: None,
                        prompt_eval_count: None,
                        prompt_eval_duration: None,
                        eval_count: None,
                        eval_duration: None,
                    };
                }
            }

            yield ChatStreamChunk {
                model: String::new(),
                created_at: String::new(),
                message: None,
                done: true,
                done_reason: Some("eof".into()),
                total_duration: None,
                load_duration: None,
                prompt_eval_count: None,
                prompt_eval_duration: None,
                eval_count: None,
                eval_duration: None,
            };
        };

        Ok(Box::pin(s))
    }


    pub async fn embeddings(&self, _req: EmbeddingsRequest) -> Result<EmbeddingsResponse, ModelClientError> {
        // As of 2025 OpenRouter does not expose an embeddings endpoint
        Err(ModelClientError::Unsupported("OpenRouter embeddings are not available".into()))
    }
}


#[derive(Serialize, Default)]
struct OrChatRequest {
    model: String,
    messages: Vec<OrMessage>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    top_a: Option<f32>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    min_p: Option<f32>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    repetition_penalty: Option<f32>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<i32>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<i32>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    logit_bias: Option<serde_json::Value>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    logprobs: Option<bool>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    top_logprobs: Option<u32>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<serde_json::Value>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    parallel_tool_calls: Option<bool>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<serde_json::Value>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    structured_outputs: Option<bool>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    verbosity: Option<String>,
}
impl From<ChatRequest> for OrChatRequest {
    fn from(value: ChatRequest) -> Self {
        let ChatRequest { base, messages, tools } = value;
        let params = OpenRouterClient::map_options(&base.options);

        let stop_vec = match base.options.as_ref().and_then(|o| o.stop.clone()) {
            Some(s) => Some(vec![s]), // until you migrate to Vec<String> in your shared model
            None => None,
        };

        Self {
            model: base.model,
            messages: OpenRouterClient::map_messages(&messages), // enhance for multimodal
            temperature: params.temperature,
            top_p: params.top_p,
            top_k: params.top_k,
            top_a: None,
            min_p: None,
            repetition_penalty: None,
            seed: None,
            max_tokens: params.max_tokens,
            presence_penalty: params.presence_penalty,
            frequency_penalty: params.frequency_penalty,
            logit_bias: None,
            logprobs: None,
            top_logprobs: None,
            stop: stop_vec,
            stream: base.stream,
            tools: tools.and_then(|t| serde_json::to_value(t).ok()),
            tool_choice: None,
            parallel_tool_calls: None,
            response_format: base.format,
            structured_outputs: None,
            verbosity: None,
        }
    }
}

#[derive(Serialize, Default)]
struct OrParams {
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    max_tokens: Option<i32>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    stop: Option<String>,
}

#[derive(Serialize, Deserialize, Clone)]
struct OrMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct OrChoice {
    message: OrMessage,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct OrChatResponse {
    _id: String,
    created: u64,
    model: String,
    choices: Vec<OrChoice>,
}

#[derive(Deserialize)]
struct OrDelta { _role: Option<String>, content: Option<String> }

#[derive(Deserialize)]
struct OrDeltaChoice { 
    delta: OrDelta, 
    _finish_reason: Option<String> 
}

#[derive(Deserialize)]
struct OrStreamChunk { 
    _id: String, 
    created: u64, 
    model: String, 
    choices: Vec<OrDeltaChoice> 
}

#[derive(Deserialize, Debug)]
struct OrErrorEnvelope {
    error: OrErrorBody,
    #[allow(dead_code)]
    user_id: Option<String>,
}

#[derive(Deserialize, Debug)]
struct OrErrorBody {
    message: String,
    code: serde_json::Value,
}

fn parse_oopen_router_error(text: &str) -> Option<ModelClientError> {
    let s = text.trim_start();
    if !s.starts_with('{') || !s.contains("\"error\"") {
        return None;
    }
    match serde_json::from_str::<OrErrorEnvelope>(s) {
        Ok(env) => {
            let code = env.error.code;
            let msg = env.error.message;
            Some(ModelClientError::Api(format!("OpenRouter error {code}: {msg}")))
        }
        Err(_) => None,
    }
}
