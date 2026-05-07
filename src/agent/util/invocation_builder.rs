use std::collections::HashMap;

use rmcp::schemars::JsonSchema;
use serde_json::Value;
use tokio::sync::mpsc::Sender;

use crate::{
    call_tools,
    services::llm::{
        message::Message, BaseRequest, ClientBuilder, InferenceOptions, ResponseFormatConfig,
        SchemaSpec,
    },
    Agent, ChatRequest, ChatResponse, ClientConfig, InvocationError, InvocationRequest,
    Notification, Provider, Tool,
};

#[derive(Debug, Clone, Default)]
pub struct InvocationBuilder {
    model: Option<String>,
    format: Option<Value>,
    stream: Option<bool>,
    keep_alive: Option<String>,

    name: Option<String>,

    // payload
    messages: Option<Vec<Message>>,
    tools: Option<Vec<Tool>>,

    // flattened options: None means inherit, Some(_) means override
    opts: InferenceOptions,
    strip_thinking: Option<bool>,
    use_tools: Option<bool>,

    /// Provider, endpoint, credentials, and headers for standalone invocations.
    client_config: ClientConfig,
    /// Notification channel to send notifications to
    notification_channel: Option<Sender<Notification>>,

    /// Response schema input plus optional provider hints.
    response_format: ResponseFormatConfig,
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
    pub fn stream(mut self, v: bool) -> Self {
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

    pub fn add_message(mut self, msg: Message) -> Self {
        self.messages.get_or_insert_with(Vec::new).push(msg);
        self
    }

    pub fn set_message(mut self, msg: Message) -> Self {
        self.messages = Some(vec![msg]);
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

    /// Set the name identifier of the invocation
    pub fn set_name<T>(mut self, name: T) -> Self
    where
        T: Into<String>,
    {
        self.name = Some(name.into());
        self
    }

    /// Select the LLM provider implementation.
    pub fn set_provider(mut self, provider: Provider) -> Self {
        self.client_config = self.client_config.provider(Some(provider));
        self
    }

    /// Override the base URL for the provider client.
    pub fn set_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.client_config = self.client_config.base_url(Some(base_url));
        self
    }

    /// Set the API key used by the provider client.
    pub fn set_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.client_config = self.client_config.api_key(Some(api_key));
        self
    }

    /// Set the organization or tenant identifier for requests.
    pub fn set_organization(mut self, organization: impl Into<String>) -> Self {
        self.client_config = self.client_config.organization(Some(organization));
        self
    }

    /// Provide additional HTTP headers to include on each request.
    pub fn set_extra_headers(mut self, extra_headers: HashMap<String, String>) -> Self {
        self.client_config = self.client_config.extra_headers(Some(extra_headers));
        self
    }

    pub fn notification_channel(
        mut self,
        notification_channel: Option<Sender<Notification>>,
    ) -> Self {
        self.notification_channel = notification_channel;
        self
    }

    // A string of JSON Schema
    pub fn set_response_format_str(mut self, schema_json: &str) -> Self {
        self.response_format.set_raw(schema_json);
        self
    }

    // A ready-made serde_json::Value
    pub fn set_response_format_value(mut self, schema: serde_json::Value) -> Self {
        self.response_format.set_value(schema);
        self
    }

    // From a Rust type via schemars
    pub fn set_response_format_from<T: JsonSchema>(mut self) -> Self {
        self.response_format.set_type::<T>();
        self
    }

    // From a Rust type via SchemaSpec
    pub fn set_response_format_spec(mut self, schema: SchemaSpec) -> Self {
        self.response_format.set_spec(schema);
        self
    }

    pub fn set_schema_name(mut self, name: impl Into<String>) -> Self {
        self.response_format.set_name(name);
        self
    }

    pub fn set_schema_strict(mut self, strict: bool) -> Self {
        self.response_format.set_strict(strict);
        self
    }

    pub async fn invoke_with(self, agent: &mut Agent) -> Result<ChatResponse, InvocationError> {
        let model = self.model.or(Some(agent.model.clone()));
        let format = match self.format {
            Some(format) => Some(format),
            None => match self
                .response_format
                .resolve()
                .map_err(InvocationError::InvalidJsonSchema)?
            {
                Some(spec) => Some(agent.inference_client.structured_output_format(&spec)?),
                None => agent.response_format.clone(),
            },
        };
        let stream = self.stream.or(Some(agent.stream));
        let keep_alive = self.keep_alive.or(agent.keep_alive.clone());
        let messages = self
            .messages
            .or(Some(agent.history.clone()))
            .unwrap_or_default();
        let tools = match self.use_tools {
            Some(false) => None,
            Some(true) | None => self.tools.or(agent.tools.clone()),
        };

        let name = self
            .name
            .or(Some(agent.name.clone()))
            .unwrap_or("Invocation".into());

        let options = self
            .opts
            .merge_over(agent.inference_options())
            .into_option();

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
            self.strip_thinking.unwrap_or(agent.strip_thinking),
            request,
            agent.inference_client.clone(),
            agent.notification_channel.clone(),
            name,
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

    pub async fn invoke(mut self) -> Result<ChatResponse, InvocationError> {
        let name = self.name.take().unwrap_or("Invocation".into());

        let options = self.opts.into_option();

        let Some(model) = self.model.take() else {
            return Err(InvocationError::ModelNotDefined);
        };

        let tools = match self.use_tools {
            Some(false) => None,
            Some(true) => self.tools.take(),
            None => self.tools.take(),
        };

        let client = self.client_config.build()?;

        let response_format = self
            .response_format
            .resolve()
            .map_err(InvocationError::InvalidJsonSchema)?;

        let format = response_format
            .map(|f| client.structured_output_format(&f))
            .transpose()?;
        let format = self.format.take().or(format);

        let request = ChatRequest {
            base: BaseRequest {
                model,
                format,
                options,
                stream: self.stream,
                keep_alive: self.keep_alive.take(),
            },
            messages: self.messages.unwrap_or_default(),
            tools,
        };

        let invcation_request = InvocationRequest::new(
            self.strip_thinking.unwrap_or(false),
            request,
            client,
            self.notification_channel.take(),
            name,
        );

        let response = match &invcation_request.request.base.stream {
            Some(true) => super::invocations::invoke_streaming(invcation_request).await?,
            _ => super::invocations::invoke_nonstreaming(invcation_request).await?,
        };

        Ok(response)
    }
}
