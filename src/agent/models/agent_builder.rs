use crate::{
    agent::models::{
        configs::{ModelConfig, PromptConfig},
        error::AgentBuildError,
    },
    notifications::Notification,
    services::{
        llm::{ClientBuilder, ClientConfig, Provider, ResponseFormatConfig, SchemaSpec},
        mcp::mcp_tool_builder::McpServerType,
    },
    skills::{build_read_skill_tool, load_skill_sources},
    templates::Template,
    Agent, Flow, FlowFuture, Skill, Tool, ToolBuilderError, SKILL_SYSTEM_PROMPT_TEMPLATE,
};
use futures::future::join_all;
use rmcp::schemars::JsonSchema;
use std::{collections::HashMap, path::PathBuf, sync::Arc};
use tokio::sync::{mpsc, Mutex};

/// A builder for [`Agent`].
///
/// Allows configuration of model, endpoint, tools, penalties, flow, etc.
/// Uses the builder pattern so you can chain calls.
///
/// Example:
///
/// ```
/// use reagent_rs::AgentBuilder;
///
/// async {
///     let mut agent = AgentBuilder::default()
///         // model must be set, everything else has
///         // defualts and is optional
///         .set_model("qwen3:0.6b")
///         .set_system_prompt("You are a helpful assistant.")
///         .set_temperature(0.6)
///         .set_num_ctx(2048)
///         // call build to return the agent
///         .build()
///         .await;
/// };
///
/// ```
///
#[derive(Debug, Default)]
pub struct AgentBuilder {
    /// Name used for logging and defaults
    name: Option<String>,

    /// Provider, endpoint, credentials, and headers for the LLM client.
    client_config: ClientConfig,
    /// Model name plus sampling/decoding options.
    model_config: ModelConfig,

    /// Optional first-message template used to build the system prompt
    template: Option<Arc<Mutex<Template>>>,
    /// Raw system prompt string seeded into history
    system_prompt: Option<String>,
    /// Local tools the agent can call during a flow
    tools: Option<Vec<Tool>>,
    /// Response schema input plus optional provider hints.
    response_format: ResponseFormatConfig,
    /// MCP tool servers the agent can reach
    mcp_servers: Option<Vec<McpServerType>>,
    /// Individual skill roots or SKILL.md files to load.
    skill_paths: Vec<PathBuf>,
    /// Directories containing multiple skill directories.
    skill_collection_paths: Vec<PathBuf>,
    /// Built-in skills loaded directly from the library.
    builtin_skills: Vec<Skill>,
    /// Prompt inserted when a tool-call branch begins
    stop_prompt: Option<String>,
    /// Stopword that indicates end of generation
    stopword: Option<String>,
    /// Whether to strip think tags from model output
    strip_thinking: Option<bool>,
    /// Safety cap on the number of conversation iterations
    max_iterations: Option<usize>,
    /// Clear conversation history before each invocation
    clear_histroy_on_invoke: Option<bool>,

    /// Enable server streaming for token events
    stream: Option<bool>,
    /// Keep-alive in memory for model after inference
    keep_alive: Option<String>,

    /// Optional mpsc sender for notifications
    notification_channel: Option<mpsc::Sender<Notification>>,
    /// High-level control flow policy
    flow: Option<Flow>,
}

impl AgentBuilder {
    /// Import generic client settings from a `ClientConfig`.
    /// Existing values already set on the builder are preserved unless overwritten by `conf`.
    /// Only fields present in `conf` are applied.
    pub fn import_client_config(mut self, conf: ClientConfig) -> Self {
        if let Some(provider) = conf.provider {
            self = self.set_provider(provider);
        }
        if let Some(base_url) = conf.base_url {
            self = self.set_base_url(base_url);
        }
        if let Some(api_key) = conf.api_key {
            self = self.set_api_key(api_key);
        }
        if let Some(organization) = conf.organization {
            self = self.set_organization(organization);
        }
        if let Some(extra_headers) = conf.extra_headers {
            self = self.set_extra_headers(extra_headers);
        }
        self
    }

    /// Import prompt-related settings from a `PromptConfig`.
    /// Existing values already set on the builder are preserved unless overwritten by `conf`.
    /// Only fields present in `conf` are applied.
    pub fn import_prompt_config(mut self, conf: PromptConfig) -> Self {
        if let Some(template) = conf.template {
            self = self.set_template(template);
        }
        if let Some(system_prompt) = conf.system_prompt {
            self = self.set_system_prompt(system_prompt);
        }
        if let Some(tools) = conf.tools {
            for tool in tools {
                self = self.add_tool(tool);
            }
        }
        if let Some(response_format) = conf.response_format {
            self = self.set_response_format_spec(response_format);
        }
        if let Some(mcp_servers) = conf.mcp_servers {
            for mcp in mcp_servers {
                self = self.add_mcp_server(mcp);
            }
        }
        if let Some(stop_prompt) = conf.stop_prompt {
            self = self.set_stop_prompt(stop_prompt);
        }
        if let Some(stopword) = conf.stopword {
            self = self.set_stopword(stopword);
        }
        if let Some(strip_thinking) = conf.strip_thinking {
            self = self.strip_thinking(strip_thinking);
        }
        if let Some(max_iterations) = conf.max_iterations {
            self = self.set_max_iterations(max_iterations);
        }
        if let Some(clear_histroy_on_invoke) = conf.clear_histroy_on_invoke {
            self = self.set_clear_history_on_invocation(clear_histroy_on_invoke);
        }

        self = self.set_stream(conf.stream);
        self
    }

    /// Import model sampling and decoding parameters from a `ModelConfig`.
    /// Existing values already set on the builder are preserved unless overwritten by `conf`.
    /// Only fields present in `conf` are applied.
    pub fn import_model_config(mut self, conf: ModelConfig) -> Self {
        if let Some(model) = conf.model {
            self = self.set_model(model)
        }
        if let Some(temperature) = conf.temperature {
            self = self.set_temperature(temperature)
        }
        if let Some(top_p) = conf.top_p {
            self = self.set_top_p(top_p)
        }
        if let Some(presence_penalty) = conf.presence_penalty {
            self = self.set_presence_penalty(presence_penalty)
        }
        if let Some(frequency_penalty) = conf.frequency_penalty {
            self = self.set_frequency_penalty(frequency_penalty)
        }
        if let Some(num_ctx) = conf.num_ctx {
            self = self.set_num_ctx(num_ctx)
        }
        if let Some(repeat_last_n) = conf.repeat_last_n {
            self = self.set_repeat_last_n(repeat_last_n)
        }
        if let Some(repeat_penalty) = conf.repeat_penalty {
            self = self.set_repeat_penalty(repeat_penalty)
        }
        if let Some(seed) = conf.seed {
            self = self.set_seed(seed)
        }
        if let Some(stop) = conf.stop {
            self = self.set_stop(stop)
        }
        if let Some(num_predict) = conf.num_predict {
            self = self.set_num_predict(num_predict)
        }
        if let Some(top_k) = conf.top_k {
            self = self.set_top_k(top_k)
        }
        if let Some(min_p) = conf.min_p {
            self = self.set_min_p(min_p)
        }

        self
    }

    /// Set the name of the agent (used in logging)
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
    pub fn set_base_url<T>(mut self, base_url: T) -> Self
    where
        T: Into<String>,
    {
        self.client_config = self.client_config.base_url(Some(base_url));
        self
    }

    /// Set the API key used by the provider client.
    pub fn set_api_key<T>(mut self, api_key: T) -> Self
    where
        T: Into<String>,
    {
        self.client_config = self.client_config.api_key(Some(api_key));
        self
    }

    /// Set the organization or tenant identifier for requests.
    pub fn set_organization<T>(mut self, organization: T) -> Self
    where
        T: Into<String>,
    {
        self.client_config = self.client_config.organization(Some(organization));
        self
    }

    /// Provide additional HTTP headers to include on each request.
    pub fn set_extra_headers(mut self, extra_headers: HashMap<String, String>) -> Self {
        self.client_config = self.client_config.extra_headers(Some(extra_headers));
        self
    }

    /// Set the streaming value for Ollam
    /// Will enable Token Notifications
    pub fn set_stream(mut self, set: bool) -> Self {
        self.stream = Some(set);
        self
    }

    /// Set the sampling temperature.
    pub fn set_temperature(mut self, v: f32) -> Self {
        self.model_config.temperature = Some(v);
        self
    }

    /// Set nucleus sampling probability.
    pub fn set_top_p(mut self, v: f32) -> Self {
        self.model_config.top_p = Some(v);
        self
    }

    /// Set presence penalty.
    pub fn set_presence_penalty(mut self, v: f32) -> Self {
        self.model_config.presence_penalty = Some(v);
        self
    }

    /// Set frequency penalty.
    pub fn set_frequency_penalty(mut self, v: f32) -> Self {
        self.model_config.frequency_penalty = Some(v);
        self
    }

    /// Set maximum context length (in tokens/chunks).
    pub fn set_num_ctx(mut self, v: u32) -> Self {
        self.model_config.num_ctx = Some(v);
        self
    }

    /// Repeat penalty for the last N tokens.
    pub fn set_repeat_last_n(mut self, v: i32) -> Self {
        self.model_config.repeat_last_n = Some(v);
        self
    }

    /// Set keep alive of the model
    pub fn set_keep_alive(mut self, v: String) -> Self {
        self.keep_alive = Some(v);
        self
    }

    /// Set penalty for repeated tokens.
    pub fn set_repeat_penalty(mut self, v: f32) -> Self {
        self.model_config.repeat_penalty = Some(v);
        self
    }

    /// Set RNG seed for sampling.
    pub fn set_seed(mut self, v: i32) -> Self {
        self.model_config.seed = Some(v);
        self
    }

    /// Set the hard stop string.
    pub fn set_stop<T: Into<String>>(mut self, v: T) -> Self {
        self.model_config.stop = Some(v.into());
        self
    }

    /// Number of tokens to predict.
    pub fn set_num_predict(mut self, v: i32) -> Self {
        self.model_config.num_predict = Some(v);
        self
    }

    /// Top-K sampling.
    pub fn set_top_k(mut self, v: u32) -> Self {
        self.model_config.top_k = Some(v);
        self
    }

    /// Minimum probability threshold.
    pub fn set_min_p(mut self, v: f32) -> Self {
        self.model_config.min_p = Some(v);
        self
    }

    /// Select the underlying model name.
    pub fn set_model<T: Into<String>>(mut self, model: T) -> Self {
        self.model_config.model = Some(model.into());
        self
    }

    /// System prompt that initializes conversation history.
    pub fn set_system_prompt<T: Into<String>>(mut self, prompt: T) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Optional prompt to insert on each tool‐call branch.
    pub fn set_stop_prompt<T: Into<String>>(mut self, stop_prompt: T) -> Self {
        self.stop_prompt = Some(stop_prompt.into());
        self
    }

    /// Optional stopword to detect end of generation.
    pub fn set_stopword<T: Into<String>>(mut self, stopword: T) -> Self {
        self.stopword = Some(stopword.into());
        self
    }

    /// Whether to strip `<think>` blocks from model output.
    pub fn strip_thinking(mut self, strip: bool) -> Self {
        self.strip_thinking = Some(strip);
        self
    }

    pub fn set_flow_fn(mut self, flow: Flow) -> Self {
        self.flow = Some(flow);
        self
    }

    pub fn set_flow<F>(self, f: F) -> Self
    where
        F: for<'a> Fn(&'a mut Agent, String) -> FlowFuture<'a> + Send + Sync + 'static,
    {
        self.set_flow_fn(Flow::from_fn(f))
    }

    /// Add a local tool.
    pub fn add_tool(mut self, tool: Tool) -> Self {
        if let Some(ref mut vec) = self.tools {
            vec.push(tool);
        } else {
            self.tools = Some(vec![tool]);
        }
        self
    }

    /// Set agent's avalible tools
    pub fn set_tools(mut self, tools: Vec<Tool>) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Remove all agents avalible tools
    pub fn remove_tools(mut self) -> Self {
        self.tools = None;
        self
    }

    /// Add an MCP server endpoint.
    pub fn add_mcp_server(mut self, server: McpServerType) -> Self {
        if let Some(ref mut svs) = self.mcp_servers {
            svs.push(server);
        } else {
            self.mcp_servers = Some(vec![server]);
        }
        self
    }

    /// Add a skill by pointing to either `SKILL.md`/`skill.md` or its containing directory.
    pub fn add_skill(mut self, path: impl Into<PathBuf>) -> Self {
        self.skill_paths.push(path.into());
        self
    }

    /// Add all skills found in the immediate child directories of a collection directory.
    pub fn add_skill_collection(mut self, path: impl Into<PathBuf>) -> Self {
        self.skill_collection_paths.push(path.into());
        self
    }

    pub fn add_bash(mut self) -> Result<Self, ToolBuilderError> {
        let bash_tool = crate::tools::prebuilt::bash::build_bash_tool(Default::default())?;

        self = self.add_tool(bash_tool);
        self = self.add_bash_skill();

        Ok(self)
    }

    fn add_bash_skill(mut self) -> Self {
        self.builtin_skills.push(crate::skills::bash_skill());
        self
    }

    /// Set a template for the agent's first prompt
    pub fn set_template(mut self, template: Template) -> Self {
        self.template = Some(Arc::new(Mutex::new(template)));
        self
    }

    /// Set max_iterations. This controlls maximum amount of times the agent
    /// may perform a "conversation iteration". Also serves as a breakpoint
    /// if the agent is stuck in a loop
    pub fn set_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = Some(max_iterations);
        self
    }

    /// if set to true, will clear the conversation histroy on each invocation
    /// of the agent
    pub fn set_clear_history_on_invocation(mut self, clear: bool) -> Self {
        self.clear_histroy_on_invoke = Some(clear);
        self
    }
    // A string of JSON Schema
    pub fn set_response_format_str(mut self, schema_json: &str) -> Self {
        self.response_format.set_raw(schema_json);
        self
    }

    /// A ready-made serde_json::Value
    pub fn set_response_format_value(mut self, schema: serde_json::Value) -> Self {
        self.response_format.set_value(schema);
        self
    }

    /// From a Rust type via schemars
    pub fn set_response_format_from<T: JsonSchema>(mut self) -> Self {
        self.response_format.set_type::<T>();
        self
    }

    /// From a Rust type via SchemaSpec
    pub fn set_response_format_spec(mut self, schema: SchemaSpec) -> Self {
        self.response_format.set_spec(schema);
        self
    }

    /// Optional hints that apply whether you used *_str, *_value, or *_from
    pub fn set_schema_name(mut self, name: impl Into<String>) -> Self {
        self.response_format.set_name(name);
        self
    }

    pub fn set_schema_strict(mut self, strict: bool) -> Self {
        self.response_format.set_strict(strict);
        self
    }

    /// Build an [`Agent`] and return also the notification receiver.
    ///
    /// Creates an internal mpsc channel of size 100.
    pub async fn build_with_notification(
        mut self,
    ) -> Result<(Agent, mpsc::Receiver<Notification>), AgentBuildError> {
        let (sender, receiver) = mpsc::channel(100);
        self.notification_channel = Some(sender);
        let agent = self.build().await?;
        Ok((agent, receiver))
    }

    /// Finalize all settings and produce an [`Agent`], or an error if required fields missing or invalid.
    pub async fn build(self) -> Result<Agent, AgentBuildError> {
        let model_config = self.model_config;
        let model = model_config
            .model
            .clone()
            .ok_or(AgentBuildError::ModelNotSet)?;

        let skill_template = Template::simple(SKILL_SYSTEM_PROMPT_TEMPLATE);

        let mut system_prompt = self
            .system_prompt
            .unwrap_or_else(|| "You are a helpful agent.".into());

        let mut skills = load_skill_sources(&self.skill_paths, &self.skill_collection_paths)?;
        skills.extend(self.builtin_skills);

        let mut tools = self.tools.clone();

        if !skills.is_empty() {
            let skill_descriptions = join_all(
                skills
                    .iter()
                    .map(|s| async move { s.discovery_description().await }),
            )
            .await;

            let skills_section = skill_descriptions.join("\n\n---\n\n");

            let data = HashMap::from([
                ("system_prompt", system_prompt),
                ("skills_discovery", skills_section),
            ]);

            system_prompt = skill_template.compile(&data).await;

            if tools
                .as_ref()
                .is_some_and(|tools| tools.iter().any(|tool| tool.name() == "read_skill"))
            {
                return Err(AgentBuildError::ReservedToolName("read_skill".into()));
            }

            let read_skill_tool = build_read_skill_tool(&skills)?;
            match tools.as_mut() {
                Some(tools) => tools.push(read_skill_tool),
                None => tools = Some(vec![read_skill_tool]),
            }
        }

        let strip_thinking = self.strip_thinking.unwrap_or(true);
        let clear_histroy_on_invoke = self.clear_histroy_on_invoke.unwrap_or(false);

        let flow = self.flow.unwrap_or(Flow::Default);

        let name = match self.name {
            Some(n) => n,
            None => format!("Agent-{model}"),
        };

        let stream = self.stream.unwrap_or(false);

        let inference_client = self.client_config.build()?;

        let response_format = self
            .response_format
            .resolve()
            .map_err(AgentBuildError::InvalidJsonSchema)?;

        let response_format = match response_format {
            Some(f) => Some(inference_client.structured_output_format(&f)?),
            None => None,
        };

        Agent::try_new(
            name,
            &model,
            inference_client,
            &system_prompt,
            tools,
            response_format,
            self.stop_prompt,
            self.stopword,
            strip_thinking,
            model_config.temperature,
            model_config.top_p,
            model_config.presence_penalty,
            model_config.frequency_penalty,
            model_config.num_ctx,
            model_config.repeat_last_n,
            model_config.repeat_penalty,
            model_config.seed,
            model_config.stop,
            model_config.num_predict,
            stream,
            model_config.top_k,
            model_config.min_p,
            self.keep_alive,
            self.notification_channel,
            self.mcp_servers,
            flow,
            self.template,
            skills,
            self.max_iterations,
            clear_histroy_on_invoke,
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use serde_json::Value;

    use super::*;
    use crate::{
        notifications::NotificationContent, Agent, AsyncToolFn, FlowFuture, Message, ToolBuilder,
    };

    #[tokio::test]
    async fn defaults_fail_without_model() {
        let err = AgentBuilder::default().build().await.unwrap_err();
        assert!(matches!(err, AgentBuildError::ModelNotSet));
    }

    #[tokio::test]
    async fn build_minimal_succeeds() {
        let agent = AgentBuilder::default()
            .set_model("test-model")
            .build()
            .await
            .expect("build should succeed");
        assert_eq!(agent.model, "test-model");
        // history initialized with system prompt
        assert_eq!(
            agent.history.len(),
            1,
            "history should contain exactly the system prompt"
        );
    }

    #[tokio::test]
    async fn custom_system_prompt_and_response_format() {
        let json = r#"{"type":"object"}"#;
        let agent = AgentBuilder::default()
            .set_model("m")
            .set_system_prompt("Hello world")
            .set_response_format_str(json)
            .build()
            .await
            .unwrap();
        assert_eq!(agent.history[0].content.as_ref().unwrap(), "Hello world");
        assert!(agent.response_format.is_some());
        assert_eq!(
            agent
                .response_format
                .as_ref()
                .unwrap()
                .get("type")
                .unwrap()
                .as_str()
                .unwrap(),
            "object"
        );
    }

    #[tokio::test]
    async fn invalid_json_schema_errors() {
        let bad = "not json";
        let err = AgentBuilder::default()
            .set_model("m")
            .set_response_format_str(bad)
            .build()
            .await
            .unwrap_err();
        assert!(matches!(err, AgentBuildError::InvalidJsonSchema(_)));
    }

    #[tokio::test]
    async fn add_tools() {
        let weather_exec: AsyncToolFn = {
            Arc::new(move |_model_args_json: Value| {
                Box::pin(async move {
                    Ok(r#"
                    {
                    "type":"object",
                    "properties":{
                        "windy":{"type":"boolean"},
                        "temperature":{"type":"integer"},
                        "description":{"type":"string"}
                    },
                    "required":["windy","temperature","description"]
                    }
                    "#
                    .into())
                })
            })
        };

        let weather_tool = ToolBuilder::new()
            .function_name("get_current_weather")
            .function_description("Returns a weather forecast for a given location")
            .add_required_property("location", "string", "City name")
            .executor(weather_exec)
            .build()
            .unwrap();

        let agent = AgentBuilder::default()
            .set_model("x")
            .add_tool(weather_tool.clone())
            .build()
            .await
            .unwrap();
        assert_eq!(agent.local_tools.unwrap()[0].name(), weather_tool.name());
    }

    #[tokio::test]
    async fn build_with_notification_channel() {
        let (agent, mut rx) = AgentBuilder::default()
            .set_model("foo")
            .build_with_notification()
            .await
            .unwrap();
        // send a notification
        agent
            .notification_channel
            .as_ref()
            .unwrap()
            .send(Notification::new(
                "test".to_string(),
                NotificationContent::Done(false, None),
            ))
            .await
            .unwrap();
        let notified = rx.recv().await.unwrap();
        assert!(matches!(
            notified.content,
            NotificationContent::Done(false, None)
        ));
    }

    #[tokio::test]
    async fn custom_flow_invocation() {
        fn echo_flow<'a>(_agent: &'a mut Agent, prompt: String) -> FlowFuture<'a> {
            Box::pin(async move { Ok(Message::system(format!("ECHO: {prompt}"))) })
        }

        let agent = AgentBuilder::default()
            .set_model("m")
            .set_flow(echo_flow)
            .build()
            .await
            .unwrap();
        let mut a = agent.clone();
        let resp = a.invoke_flow("abc").await.unwrap();
        assert_eq!(resp.content.unwrap(), "ECHO: abc");
    }
}
