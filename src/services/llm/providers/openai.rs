use futures::{Stream, StreamExt};
use reqwest::{
    header::{HeaderMap, HeaderName, HeaderValue, AUTHORIZATION, CONTENT_TYPE},
    Client,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::BTreeMap, pin::Pin};
use tracing::{debug, instrument};

use crate::{
    services::llm::{
        message::Message,
        models::{
            base::{InferenceOptions, Role},
            chat::{ChatRequest, ChatResponse, ChatStreamChunk},
            embedding::{EmbeddingsRequest, EmbeddingsResponse},
            errors::InferenceClientError,
        },
        StructuredOuputFormat,
    },
    ClientConfig, Tool, ToolCall, ToolCallFunction, ToolType,
};

#[derive(Debug, Clone)]
pub struct OpenAiClient {
    client: Client,
    base_url: String,
}

impl OpenAiClient {
    pub fn new(cfg: ClientConfig) -> Result<Self, InferenceClientError> {
        let base_url = cfg
            .base_url
            .unwrap_or_else(|| "https://api.openai.com/v1".to_string());

        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        if let Some(api_key) = cfg.api_key {
            headers.insert(
                AUTHORIZATION,
                HeaderValue::from_str(&format!("Bearer {api_key}")).map_err(|e| {
                    InferenceClientError::Config(format!("Invalid api_key header: {e}"))
                })?,
            );
        }

        if let Some(organization) = cfg.organization {
            headers.insert(
                HeaderName::from_static("openai-organization"),
                HeaderValue::from_str(&organization).map_err(|e| {
                    InferenceClientError::Config(format!("Invalid organization header: {e}"))
                })?,
            );
        }

        if let Some(extra) = cfg.extra_headers {
            for (k, v) in extra.into_iter() {
                let name = HeaderName::from_bytes(k.as_bytes()).map_err(|_| {
                    InferenceClientError::Config(format!("Invalid header name: {k}"))
                })?;
                let value = HeaderValue::from_str(&v).map_err(|_| {
                    InferenceClientError::Config(format!("Invalid header value for {k}"))
                })?;
                headers.insert(name, value);
            }
        }

        let client = Client::builder().default_headers(headers).build()?;

        Ok(Self { client, base_url })
    }

    fn endpoint_url(&self, endpoint: &str) -> String {
        format!(
            "{}/{}",
            self.base_url.trim_end_matches('/'),
            endpoint.trim_start_matches('/')
        )
    }

    fn map_messages(messages: &[Message]) -> Vec<OpenAiMessage> {
        messages.iter().map(OpenAiMessage::from).collect()
    }

    fn map_options(options: &Option<InferenceOptions>) -> OpenAiParams {
        let mut params = OpenAiParams::default();
        if let Some(options) = options {
            params.temperature = options.temperature;
            params.top_p = options.top_p;
            params.max_tokens = options.max_tokens.or(options.num_predict);
            params.presence_penalty = options.presence_penalty;
            params.frequency_penalty = options.frequency_penalty;
            params.stop = options.stop.clone();
            params.seed = options.seed;
        }
        params
    }

    async fn post_json<B: Serialize>(
        &self,
        endpoint: &str,
        body: &B,
    ) -> Result<String, InferenceClientError> {
        let resp = self
            .client
            .post(self.endpoint_url(endpoint))
            .json(body)
            .send()
            .await?;
        let status = resp.status();
        let text = resp.text().await?;

        if !status.is_success() {
            if let Some(e) = parse_openai_error(&text) {
                return Err(e);
            }
            return Err(InferenceClientError::Api(format!(
                "Request failed: {status} - {text}"
            )));
        }

        if let Some(e) = parse_openai_error(&text) {
            return Err(e);
        }

        Ok(text)
    }

    #[instrument(name = "openai.chat", skip_all)]
    async fn chat_inner(
        &self,
        req: ChatRequest,
        stream: bool,
    ) -> Result<reqwest::Response, InferenceClientError> {
        let mut body = OpenAiChatRequest::from(req);
        body.stream = Some(stream);
        let resp = self
            .client
            .post(self.endpoint_url("/chat/completions"))
            .json(&body)
            .send()
            .await?;
        Ok(resp)
    }

    pub async fn chat(&self, req: ChatRequest) -> Result<ChatResponse, InferenceClientError> {
        let mut body = OpenAiChatRequest::from(req);
        body.stream = Some(false);
        let text = self.post_json("/chat/completions", &body).await?;

        let response: OpenAiChatResponse = serde_json::from_str(&text).map_err(|e| {
            InferenceClientError::Serialization(format!("decode error: {e}; raw: {text}"))
        })?;

        let choice = response.choices.into_iter().next();
        let done_reason = choice
            .as_ref()
            .and_then(|choice| choice.finish_reason.clone());
        let message = choice
            .map(|choice| message_from_openai(choice.message))
            .unwrap_or_else(|| Message::assistant(String::new()));

        Ok(ChatResponse {
            model: response.model,
            created_at: response.created.unwrap_or_default().to_string(),
            message,
            done: true,
            done_reason,
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
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<ChatStreamChunk, InferenceClientError>> + Send + 'static>>,
        InferenceClientError,
    > {
        use async_stream::try_stream;

        let resp = self.chat_inner(req, true).await?;
        let status = resp.status();

        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            if let Some(e) = parse_openai_error(&text) {
                return Err(e);
            }
            return Err(InferenceClientError::Api(format!(
                "Request failed: {status} - {text}"
            )));
        }

        let byte_stream = resp.bytes_stream();
        let s = try_stream! {
            let mut buf = Vec::<u8>::new();
            let mut partial_tool_calls = BTreeMap::<usize, PartialToolCall>::new();
            let mut latest_model = String::new();
            let mut latest_created = String::new();
            let mut done_reason: Option<String> = None;
            futures::pin_mut!(byte_stream);

            while let Some(chunk) = byte_stream.next().await {
                let chunk = chunk.map_err(|e| InferenceClientError::Request(e.to_string()))?;
                buf.extend_from_slice(&chunk);

                while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
                    let line: Vec<u8> = buf.drain(..=pos).collect();
                    let line = String::from_utf8_lossy(&line).trim().to_string();

                    if line.is_empty() || line.starts_with(':') {
                        continue;
                    }
                    if !line.starts_with("data:") {
                        continue;
                    }

                    let data = line[5..].trim();

                    if data == "[DONE]" {
                        if !partial_tool_calls.is_empty() {
                            let mut msg = Message::assistant(String::new());
                            msg.content = None;
                            msg.tool_calls = Some(finalize_tool_calls(&partial_tool_calls));
                            yield ChatStreamChunk {
                                model: latest_model.clone(),
                                created_at: latest_created.clone(),
                                message: Some(msg),
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

                        yield done_stream_chunk(
                            latest_model,
                            latest_created,
                            done_reason.or_else(|| Some("stop".into())),
                        );
                        return;
                    }

                    if let Some(e) = parse_openai_error(data) {
                        Err(e)?;
                    }

                    let parsed: OpenAiStreamChunk = match serde_json::from_str(data) {
                        Ok(value) => value,
                        Err(e) => {
                            debug!(err = %e, raw = %data, "stream json decode error");
                            continue;
                        }
                    };

                    latest_model = parsed.model.unwrap_or_else(|| latest_model.clone());
                    if let Some(created) = parsed.created {
                        latest_created = created.to_string();
                    }

                    for choice in parsed.choices {
                        if let Some(reason) = choice.finish_reason {
                            done_reason = Some(reason);
                        }

                        if let Some(content) = choice.delta.content {
                            yield ChatStreamChunk {
                                model: latest_model.clone(),
                                created_at: latest_created.clone(),
                                message: Some(Message::assistant(content)),
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

                        if let Some(tool_calls) = choice.delta.tool_calls {
                            merge_tool_call_deltas(&mut partial_tool_calls, tool_calls);
                        }
                    }
                }
            }

            if !partial_tool_calls.is_empty() {
                let mut msg = Message::assistant(String::new());
                msg.content = None;
                msg.tool_calls = Some(finalize_tool_calls(&partial_tool_calls));
                yield ChatStreamChunk {
                    model: latest_model.clone(),
                    created_at: latest_created.clone(),
                    message: Some(msg),
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

            yield done_stream_chunk(latest_model, latest_created, done_reason.or_else(|| Some("eof".into())));
        };

        Ok(Box::pin(s))
    }

    pub async fn embeddings(
        &self,
        req: EmbeddingsRequest,
    ) -> Result<EmbeddingsResponse, InferenceClientError> {
        let body = OpenAiEmbeddingsRequest {
            model: req.model,
            input: req.input,
        };
        let text = self.post_json("/embeddings", &body).await?;
        let response: OpenAiEmbeddingsResponse = serde_json::from_str(&text).map_err(|e| {
            InferenceClientError::Serialization(format!("decode error: {e}; raw: {text}"))
        })?;

        let Some(first) = response.data.into_iter().next() else {
            return Err(InferenceClientError::Api(
                "OpenAI embeddings response did not include data".into(),
            ));
        };

        Ok(EmbeddingsResponse {
            embedding: first.embedding,
        })
    }
}

#[derive(Serialize)]
struct OpenAiChatRequest {
    model: String,
    messages: Vec<OpenAiMessage>,

    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<i32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<i32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<Tool>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<Value>,
}

impl From<ChatRequest> for OpenAiChatRequest {
    fn from(value: ChatRequest) -> Self {
        let ChatRequest {
            base,
            messages,
            tools,
        } = value;
        let params = OpenAiClient::map_options(&base.options);

        Self {
            model: base.model,
            messages: OpenAiClient::map_messages(&messages),
            temperature: params.temperature,
            top_p: params.top_p,
            max_tokens: params.max_tokens,
            presence_penalty: params.presence_penalty,
            frequency_penalty: params.frequency_penalty,
            stop: params.stop,
            seed: params.seed,
            stream: base.stream,
            tools,
            response_format: base.format,
        }
    }
}

#[derive(Default)]
struct OpenAiParams {
    temperature: Option<f32>,
    top_p: Option<f32>,
    max_tokens: Option<i32>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    stop: Option<String>,
    seed: Option<i32>,
}

#[derive(Serialize, Deserialize, Clone)]
struct OpenAiMessage {
    role: String,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    content: Option<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenAiToolCall>>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

impl From<&Message> for OpenAiMessage {
    fn from(message: &Message) -> Self {
        Self {
            role: match message.role {
                Role::System => "system".to_string(),
                Role::Developer => "system".to_string(),
                Role::User => "user".to_string(),
                Role::Assistant => "assistant".to_string(),
                Role::Tool => "tool".to_string(),
            },
            content: message.content.clone(),
            tool_calls: message
                .tool_calls
                .as_ref()
                .map(|calls| calls.iter().map(OpenAiToolCall::from).collect()),
            tool_call_id: message.tool_call_id.clone(),
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
struct OpenAiToolCall {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    id: Option<String>,

    #[serde(rename = "type", default = "function_tool_type")]
    tool_type: String,

    function: OpenAiToolCallFunction,
}

impl From<&ToolCall> for OpenAiToolCall {
    fn from(call: &ToolCall) -> Self {
        Self {
            id: call.id.clone(),
            tool_type: "function".to_string(),
            function: OpenAiToolCallFunction {
                name: call.function.name.clone(),
                arguments: serde_json::to_string(&call.function.arguments)
                    .unwrap_or_else(|_| call.function.arguments.to_string()),
            },
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
struct OpenAiToolCallFunction {
    name: String,
    arguments: String,
}

#[derive(Deserialize)]
struct OpenAiChatResponse {
    #[serde(rename = "id")]
    _id: Option<String>,
    #[serde(rename = "object")]
    _object: Option<String>,
    created: Option<u64>,
    model: String,
    choices: Vec<OpenAiChoice>,
}

#[derive(Deserialize)]
struct OpenAiChoice {
    message: OpenAiResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct OpenAiResponseMessage {
    #[serde(rename = "role")]
    _role: Option<String>,
    content: Option<String>,
    tool_calls: Option<Vec<OpenAiToolCall>>,
}

#[derive(Deserialize)]
struct OpenAiStreamChunk {
    #[serde(rename = "id")]
    _id: Option<String>,
    #[serde(rename = "object")]
    _object: Option<String>,
    created: Option<u64>,
    model: Option<String>,
    choices: Vec<OpenAiStreamChoice>,
}

#[derive(Deserialize)]
struct OpenAiStreamChoice {
    delta: OpenAiStreamDelta,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct OpenAiStreamDelta {
    #[serde(rename = "role")]
    _role: Option<String>,
    content: Option<String>,
    tool_calls: Option<Vec<OpenAiStreamToolCall>>,
}

#[derive(Deserialize)]
struct OpenAiStreamToolCall {
    index: usize,
    id: Option<String>,
    #[serde(rename = "type")]
    _tool_type: Option<String>,
    function: Option<OpenAiStreamToolCallFunction>,
}

#[derive(Deserialize)]
struct OpenAiStreamToolCallFunction {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Default)]
struct PartialToolCall {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
}

#[derive(Serialize)]
struct OpenAiEmbeddingsRequest {
    model: String,
    input: String,
}

#[derive(Deserialize)]
struct OpenAiEmbeddingsResponse {
    data: Vec<OpenAiEmbeddingData>,
}

#[derive(Deserialize)]
struct OpenAiEmbeddingData {
    embedding: Vec<f64>,
}

#[derive(Deserialize, Debug)]
struct OpenAiErrorEnvelope {
    error: OpenAiErrorBody,
}

#[derive(Deserialize, Debug)]
struct OpenAiErrorBody {
    message: String,
    #[serde(rename = "type")]
    _error_type: Option<String>,
    code: Option<Value>,
    #[allow(dead_code)]
    param: Option<String>,
}

fn function_tool_type() -> String {
    "function".to_string()
}

fn message_from_openai(message: OpenAiResponseMessage) -> Message {
    let mut out = Message::assistant(message.content.clone().unwrap_or_default());
    out.content = message.content;
    out.tool_calls = message.tool_calls.and_then(|calls| {
        if calls.is_empty() {
            None
        } else {
            Some(calls.into_iter().map(tool_call_from_openai).collect())
        }
    });
    out
}

fn tool_call_from_openai(call: OpenAiToolCall) -> ToolCall {
    ToolCall {
        id: call.id,
        tool_type: ToolType::Function,
        function: ToolCallFunction {
            name: call.function.name,
            arguments: parse_tool_arguments(&call.function.arguments),
        },
    }
}

fn merge_tool_call_deltas(
    partials: &mut BTreeMap<usize, PartialToolCall>,
    deltas: Vec<OpenAiStreamToolCall>,
) {
    for delta in deltas {
        let partial = partials.entry(delta.index).or_default();
        if let Some(id) = delta.id {
            partial.id = Some(id);
        }
        if let Some(function) = delta.function {
            if let Some(name) = function.name {
                partial.name = Some(name);
            }
            if let Some(arguments) = function.arguments {
                partial.arguments.push_str(&arguments);
            }
        }
    }
}

fn finalize_tool_calls(partials: &BTreeMap<usize, PartialToolCall>) -> Vec<ToolCall> {
    partials
        .values()
        .map(|partial| ToolCall {
            id: partial.id.clone(),
            tool_type: ToolType::Function,
            function: ToolCallFunction {
                name: partial.name.clone().unwrap_or_default(),
                arguments: parse_tool_arguments(&partial.arguments),
            },
        })
        .collect()
}

fn parse_tool_arguments(arguments: &str) -> Value {
    if arguments.trim().is_empty() {
        return Value::Object(Default::default());
    }
    serde_json::from_str(arguments).unwrap_or_else(|_| Value::String(arguments.to_string()))
}

fn done_stream_chunk(
    model: String,
    created_at: String,
    done_reason: Option<String>,
) -> ChatStreamChunk {
    ChatStreamChunk {
        model,
        created_at,
        message: None,
        done: true,
        done_reason,
        total_duration: None,
        load_duration: None,
        prompt_eval_count: None,
        prompt_eval_duration: None,
        eval_count: None,
        eval_duration: None,
    }
}

fn parse_openai_error(text: &str) -> Option<InferenceClientError> {
    let text = text.trim_start();
    if !text.starts_with('{') || !text.contains("\"error\"") {
        return None;
    }

    match serde_json::from_str::<OpenAiErrorEnvelope>(text) {
        Ok(error) => {
            let code = error
                .error
                .code
                .map(|code| format!("{code}: "))
                .unwrap_or_default();
            Some(InferenceClientError::Api(format!(
                "OpenAI error {code}{}",
                error.error.message
            )))
        }
        Err(_) => None,
    }
}

impl StructuredOuputFormat for OpenAiClient {
    fn format(spec: &crate::services::llm::SchemaSpec) -> Value {
        serde_json::json!({
            "type": "json_schema",
            "json_schema": {
                "name": spec.name.clone().unwrap_or_else(|| "schema".to_string()),
                "strict": spec.strict.unwrap_or(false),
                "schema": spec.schema
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::llm::models::base::BaseRequest;

    #[test]
    fn openai_client_allows_local_endpoint_without_api_key() {
        let client = OpenAiClient::new(ClientConfig {
            base_url: Some("http://localhost:8000/v1".into()),
            ..Default::default()
        })
        .unwrap();

        assert_eq!(client.base_url, "http://localhost:8000/v1");
    }

    #[test]
    fn chat_request_uses_openai_compatible_shape() {
        let request = ChatRequest {
            base: BaseRequest {
                model: "DeepSeek-V4-Flash".into(),
                format: None,
                options: Some(InferenceOptions {
                    temperature: Some(0.2),
                    top_k: Some(40),
                    num_predict: Some(128),
                    ..Default::default()
                }),
                stream: Some(false),
                keep_alive: Some("1m".into()),
            },
            messages: vec![Message::system("Be concise."), Message::user("Say hi.")],
            tools: None,
        };

        let body = serde_json::to_value(OpenAiChatRequest::from(request)).unwrap();

        assert_eq!(body["model"], "DeepSeek-V4-Flash");
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["max_tokens"], 128);
        assert!(body.get("top_k").is_none());
        assert!(body.get("keep_alive").is_none());
    }

    #[test]
    fn response_tool_arguments_are_parsed_as_json() {
        let call = OpenAiToolCall {
            id: Some("call_1".into()),
            tool_type: "function".into(),
            function: OpenAiToolCallFunction {
                name: "read_skill".into(),
                arguments: r#"{"name":"summarizer"}"#.into(),
            },
        };

        let tool_call = tool_call_from_openai(call);

        assert_eq!(tool_call.function.name, "read_skill");
        assert_eq!(tool_call.function.arguments["name"], "summarizer");
    }

    #[test]
    fn empty_response_tool_calls_are_ignored() {
        let message = message_from_openai(OpenAiResponseMessage {
            _role: Some("assistant".into()),
            content: Some("No tools needed.".into()),
            tool_calls: Some(Vec::new()),
        });

        assert_eq!(message.content.as_deref(), Some("No tools needed."));
        assert!(message.tool_calls.is_none());
    }
}
