use std::collections::HashMap;

use serde_json::Value;
use tokio::sync::mpsc::Sender;

use crate::{
    call_tools,
    services::llm::{BaseRequest, ClientBuilder, InferenceOptions},
    Agent, ChatRequest, ChatResponse, ClientConfig, InvocationError, InvocationRequest, Message,
    Notification, Provider, Tool,
};

#[derive(Debug, Clone, Default)]
pub struct InvocationBuilder {
    model: Option<String>,
    format: Option<Value>,
    stream: Option<bool>,
    keep_alive: Option<String>,

    // payload
    messages: Option<Vec<Message>>,
    tools: Option<Vec<Tool>>,

    // flattened options: None means inherit, Some(_) means override
    opts: InferenceOptions,
    strip_thinking: Option<bool>,
    use_tools: Option<bool>,

    /// Provider selection for the LLM client
    provider: Option<Provider>,
    /// Optional base URL for custom or self-hosted endpoints
    base_url: Option<String>,
    /// API key used by the selected provider
    api_key: Option<String>,
    /// Optional organization or tenant identifier
    organization: Option<String>,
    /// Extra HTTP headers appended to every request
    extra_headers: Option<HashMap<String, String>>,
    /// Notification channel to send notifications to
    notification_channel: Option<Sender<Notification>>,
}

impl InvocationBuilder {
    pub fn model(mut self, v: impl Into<String>) -> Self {
        self.model = Some(v.into());
        self
    }
    pub fn response_format_some(mut self, v: Value) -> Self {
        self.format = Some(v);
        self
    }
    pub fn steam(mut self, v: bool) -> Self {
        self.stream = Some(v);
        self
    }
    pub fn keep_alive(mut self, v: impl Into<String>) -> Self {
        self.keep_alive = Some(v.into());
        self
    }
    pub fn messages(mut self, msgs: Vec<Message>) -> Self {
        self.messages = Some(msgs);
        self
    }
    pub fn history(mut self, msg: Vec<Message>) -> Self {
        self.messages = Some(msg);
        self
    }
    pub fn tools(mut self, tools: Vec<Tool>) -> Self {
        self.tools = Some(tools);
        self
    }
    pub fn add_tool(mut self, tools: Vec<Tool>) -> Self {
        self.tools = Some(tools);
        self
    }
    pub fn num_ctx(mut self, v: u32) -> Self {
        self.opts.num_ctx = Some(v);
        self
    }
    pub fn repeat_last_n(mut self, v: i32) -> Self {
        self.opts.repeat_last_n = Some(v);
        self
    }
    pub fn repeat_penalty(mut self, v: f32) -> Self {
        self.opts.repeat_penalty = Some(v);
        self
    }
    pub fn temperature(mut self, v: f32) -> Self {
        self.opts.temperature = Some(v);
        self
    }
    pub fn seed(mut self, v: i32) -> Self {
        self.opts.seed = Some(v);
        self
    }
    pub fn stop(mut self, v: String) -> Self {
        self.opts.stop = Some(v);
        self
    }
    pub fn num_predict(mut self, v: i32) -> Self {
        self.opts.num_predict = Some(v);
        self
    }
    pub fn top_k(mut self, v: u32) -> Self {
        self.opts.top_k = Some(v);
        self
    }
    pub fn top_p(mut self, v: f32) -> Self {
        self.opts.top_p = Some(v);
        self
    }
    pub fn min_p(mut self, v: f32) -> Self {
        self.opts.min_p = Some(v);
        self
    }
    pub fn presence_penalty(mut self, v: f32) -> Self {
        self.opts.presence_penalty = Some(v);
        self
    }
    pub fn frequency_penalty(mut self, v: f32) -> Self {
        self.opts.frequency_penalty = Some(v);
        self
    }
    pub fn max_tokens(mut self, v: i32) -> Self {
        self.opts.max_tokens = Some(v);
        self
    }
    pub fn strip_thinking(mut self, strip_thinking: bool) -> Self {
        self.strip_thinking = Some(strip_thinking);
        self
    }
    pub fn use_tools(mut self, use_tools: bool) -> Self {
        self.use_tools = Some(use_tools);
        self
    }
    /// Select the LLM provider implementation.
    pub fn set_provider(mut self, provider: Provider) -> Self {
        self.provider = Some(provider);
        self
    }

    /// Override the base URL for the provider client.
    pub fn set_base_url<T>(mut self, base_url: T) -> Self
    where
        T: Into<String>,
    {
        self.base_url = Some(base_url.into());
        self
    }

    /// Set the API key used by the provider client.
    pub fn set_api_key<T>(mut self, api_key: T) -> Self
    where
        T: Into<String>,
    {
        self.api_key = Some(api_key.into());
        self
    }

    /// Set the organization or tenant identifier for requests.
    pub fn set_organization<T>(mut self, organization: T) -> Self
    where
        T: Into<String>,
    {
        self.organization = Some(organization.into());
        self
    }

    /// Provide additional HTTP headers to include on each request.
    pub fn set_extra_headers(mut self, extra_headers: HashMap<String, String>) -> Self {
        self.extra_headers = Some(extra_headers);
        self
    }

    pub fn notification_channel(
        mut self,
        notification_channel: Option<Sender<Notification>>,
    ) -> Self {
        self.notification_channel = notification_channel;
        self
    }

    pub async fn invoke_with(self, agent: &mut Agent) -> Result<ChatResponse, InvocationError> {
        let model = self.model.or(Some(agent.model.clone()));
        let format = self.format.or(agent.response_format.clone());
        let stream = self.stream.or(Some(agent.stream));
        let keep_alive = self.keep_alive.or(agent.keep_alive.clone());
        let messages = self
            .messages
            .or(Some(agent.history.clone()))
            .unwrap_or(vec![]);
        let tools = self.tools.or(agent.tools.clone());

        // merge inference options field by field
        let merged_opts = InferenceOptions {
            num_ctx: self.opts.num_ctx.or(agent.num_ctx),
            repeat_last_n: self.opts.repeat_last_n.or(agent.repeat_last_n),
            repeat_penalty: self.opts.repeat_penalty.or(agent.repeat_penalty),
            temperature: self.opts.temperature.or(agent.temperature),
            seed: self.opts.seed.or(agent.seed),
            stop: self.opts.stop.or_else(|| agent.stop.clone()),
            num_predict: self.opts.num_predict.or(agent.num_predict),
            top_k: self.opts.top_k.or(agent.top_k),
            top_p: self.opts.top_p.or(agent.top_p),
            min_p: self.opts.min_p.or(agent.min_p),
            presence_penalty: self.opts.presence_penalty.or(agent.presence_penalty),
            frequency_penalty: self.opts.frequency_penalty.or(agent.frequency_penalty),
            max_tokens: self.opts.max_tokens.or(None),
        };

        let options = if all_none(&merged_opts) {
            None
        } else {
            Some(merged_opts)
        };

        let Some(model) = model else {
            return Err(InvocationError::ModelNotDefined);
        };

        let request = ChatRequest {
            base: BaseRequest {
                model,
                format,
                options,
                stream,
                keep_alive,
            },
            messages,
            tools,
        };

        let invcation_request = InvocationRequest::new(
            self.strip_thinking.unwrap_or(false),
            request,
            agent.inference_client.clone(),
            self.notification_channel,
        );

        let response = match &invcation_request.request.base.stream {
            Some(true) => super::invocations::invoke_streaming(invcation_request).await?,
            _ => super::invocations::invoke_nonstreaming(invcation_request).await?,
        };

        agent.history.push(response.message.clone());

        if let Some(tc) = response.message.tool_calls.clone() {
            for tool_msg in call_tools(agent, &tc).await {
                agent.history.push(tool_msg);
            }
        }

        Ok(response)
    }

    pub async fn invoke(self) -> Result<ChatResponse, InvocationError> {
        // merge inference options field by field
        let merged_opts = InferenceOptions {
            num_ctx: self.opts.num_ctx,
            repeat_last_n: self.opts.repeat_last_n,
            repeat_penalty: self.opts.repeat_penalty,
            temperature: self.opts.temperature,
            seed: self.opts.seed,
            stop: self.opts.stop,
            num_predict: self.opts.num_predict,
            top_k: self.opts.top_k,
            top_p: self.opts.top_p,
            min_p: self.opts.min_p,
            presence_penalty: self.opts.presence_penalty,
            frequency_penalty: self.opts.frequency_penalty,
            max_tokens: self.opts.max_tokens.or(None),
        };

        let options = if all_none(&merged_opts) {
            None
        } else {
            Some(merged_opts)
        };

        let Some(model) = self.model else {
            return Err(InvocationError::ModelNotDefined);
        };

        let tools = match self.use_tools {
            Some(false) => None,
            Some(true) => self.tools,
            None => self.tools,
        };

        let request = ChatRequest {
            base: BaseRequest {
                model,
                format: self.format,
                options,
                stream: self.stream,
                keep_alive: self.keep_alive,
            },
            messages: self.messages.unwrap_or(vec![]),
            tools: tools,
        };

        let client = ClientConfig::default()
            .provider(self.provider)
            .base_url(self.base_url)
            .api_key(self.api_key)
            .organization(self.organization)
            .extra_headers(self.extra_headers)
            .build()?;

        let invcation_request = InvocationRequest::new(
            self.strip_thinking.unwrap_or(false),
            request,
            client,
            self.notification_channel,
        );

        let response = match &invcation_request.request.base.stream {
            Some(true) => super::invocations::invoke_streaming(invcation_request).await?,
            _ => super::invocations::invoke_nonstreaming(invcation_request).await?,
        };

        Ok(response)
    }
}

fn all_none(o: &InferenceOptions) -> bool {
    o.num_ctx.is_none()
        && o.repeat_last_n.is_none()
        && o.repeat_penalty.is_none()
        && o.temperature.is_none()
        && o.seed.is_none()
        && o.stop.is_none()
        && o.num_predict.is_none()
        && o.top_k.is_none()
        && o.top_p.is_none()
        && o.min_p.is_none()
        && o.presence_penalty.is_none()
        && o.frequency_penalty.is_none()
        && o.max_tokens.is_none()
}
