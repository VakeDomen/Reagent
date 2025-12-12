use std::collections::HashMap;

use rmcp::schemars::{gen::SchemaSettings, schema::RootSchema, JsonSchema, SchemaGenerator};
use serde_json::Value;
use tokio::sync::mpsc::Sender;

use crate::{
    call_tools,
    services::llm::{BaseRequest, ClientBuilder, InferenceOptions, SchemaSpec},
    Agent, ChatRequest, ChatResponse, ClientConfig, InvocationError, InvocationRequest, Message,
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

    // The normalized, typed form used by Agent and provider adapters
    response_format: Option<SchemaSpec>,
    // Optional raw JSON string the user gave; parsed and merged at build
    response_format_raw: Option<String>,
    // Optional hint when caller set only a raw string
    pending_name: Option<String>,
    // Optional hint when caller set only a raw string
    pending_strict: Option<bool>,
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
        self.provider = Some(provider);
        self
    }

    /// Override the base URL for the provider client.
    pub fn set_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// Set the API key used by the provider client.
    pub fn set_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Set the organization or tenant identifier for requests.
    pub fn set_organization(mut self, organization: impl Into<String>) -> Self {
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

    // A string of JSON Schema
    pub fn set_response_format_str(mut self, schema_json: &str) -> Self {
        self.response_format_raw = Some(schema_json.to_owned());
        self
    }

    // A ready-made serde_json::Value
    pub fn set_response_format_value(mut self, schema: serde_json::Value) -> Self {
        self.response_format = Some(SchemaSpec {
            schema,
            name: None,
            strict: None,
        });
        self
    }

    // From a Rust type via schemars
    pub fn set_response_format_from<T: JsonSchema>(mut self) -> Self {
        let settings = SchemaSettings::draft07().with(|s| {
            s.inline_subschemas = true;
            s.meta_schema = None;
        });
        let gen = SchemaGenerator::new(settings);
        let root: RootSchema = gen.into_root_schema_for::<T>();
        let mut schema = serde_json::to_value(&root.schema).unwrap();
        if let Some(obj) = schema.as_object_mut() {
            obj.remove("$schema");
            obj.remove("definitions");
        }
        self.response_format = Some(SchemaSpec {
            schema,
            name: None,
            strict: None,
        });
        self
    }

    // From a Rust type via SchemaSpec
    pub fn set_response_format_spec(mut self, schema: SchemaSpec) -> Self {
        self.response_format = Some(schema);
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
            .unwrap_or_default();
        let tools = self.tools.or(agent.tools.clone());

        let name = self
            .name
            .or(Some(agent.name.clone()))
            .unwrap_or("Invocation".into());

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
        // merge inference options field by field
        let merged_opts = InferenceOptions {
            num_ctx: self.opts.num_ctx,
            repeat_last_n: self.opts.repeat_last_n,
            repeat_penalty: self.opts.repeat_penalty,
            temperature: self.opts.temperature,
            seed: self.opts.seed,
            stop: self.opts.stop.take(),
            num_predict: self.opts.num_predict,
            top_k: self.opts.top_k,
            top_p: self.opts.top_p,
            min_p: self.opts.min_p,
            presence_penalty: self.opts.presence_penalty,
            frequency_penalty: self.opts.frequency_penalty,
            max_tokens: self.opts.max_tokens.or(None),
        };

        let name = self.name.take().unwrap_or("Invocation".into());

        let options = if all_none(&merged_opts) {
            None
        } else {
            Some(merged_opts)
        };

        let Some(model) = self.model.take() else {
            return Err(InvocationError::ModelNotDefined);
        };

        let tools = match self.use_tools {
            Some(false) => None,
            Some(true) => self.tools.take(),
            None => self.tools.take(),
        };

        let client = ClientConfig::default()
            .provider(self.provider.take())
            .base_url(self.base_url.take())
            .api_key(self.api_key.take())
            .organization(self.organization.take())
            .extra_headers(self.extra_headers.take())
            .build()?;

        if self.response_format.is_some() && self.response_format_raw.is_some() {
            return Err(InvocationError::InvalidJsonSchema(
                "Both set_structured_output_* and \
                set_response_format_str were called. Use only one source."
                    .to_string(),
            ));
        }

        let response_format: Option<SchemaSpec> = if let Some(spec) = self.response_format.take() {
            Some(SchemaSpec {
                name: self.pending_name.take().or(spec.name),
                strict: self.pending_strict.or(spec.strict),
                schema: spec.schema,
            })
        } else if let Some(raw) = self.response_format_raw.take() {
            let v: serde_json::Value = serde_json::from_str(raw.trim()).map_err(|e| {
                InvocationError::InvalidJsonSchema(format!("Failed to parse JSON schema: {e}"))
            })?;
            Some(SchemaSpec {
                schema: v,
                name: self.pending_name.take(),
                strict: self.pending_strict,
            })
        } else {
            None
        };

        let format = response_format
            .map(|f| client.structured_output_format(&f))
            .transpose()?;

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
