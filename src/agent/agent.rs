use crate::agent::config::PromptConfig;
use crate::agent::error::{AgentBuildError, AgentError};
use crate::services::llm::{ClientConfig, SchemaSpec};
use crate::skills::Skill;
use crate::templates::Template;
use crate::{default_flow, Flow, InferenceOptions, LlmModel, NotificationHandler, Standard};
use core::fmt;
use opentelemetry::trace::TraceContextExt;
use serde::de::DeserializeOwned;
use serde_json::{Error, Value};
use std::{collections::HashMap, fs, path::Path};
use std::{marker::PhantomData, sync::Arc};
use tokio::sync::mpsc::{self, Sender};
use tokio::sync::Mutex;
use tracing::{span, Instrument, Level};
use tracing_opentelemetry::OpenTelemetrySpanExt;

use crate::{
    notifications::Notification,
    services::{llm::models::message::Message, mcp::mcp_tool_builder::get_mcp_tools},
    McpServerType, Tool,
};

#[derive(Debug, Clone, Default)]
pub struct Prompt;

#[derive(Debug, Clone, Default)]
pub struct TemplateInput;

#[derive(Clone)]
pub struct Agent<I = Prompt, O = Standard> {
    /// Human-readable name of the agent.
    pub name: String,
    /// Reusable, sessionless model used for inference.
    pub model: LlmModel,
    /// Conversation history with the model.
    pub history: Vec<Message>,
    /// Locally registered tools (before MCP merge).
    pub local_tools: Option<Vec<Tool>>,
    /// Configured MCP server endpoints.
    pub mcp_servers: Option<Vec<McpServerType>>,
    /// Fully compiled tool set (local + MCP).
    pub tools: Option<Vec<Tool>>,
    /// Provider-neutral JSON schema for responses, if any.
    pub response_format: Option<SchemaSpec>,
    /// System prompt injected at the start of the conversation.
    pub system_prompt: String,
    /// Optional stop prompt inserted on tool branches.
    pub stop_prompt: Option<String>,
    /// Stopword to detect end of generation.
    pub stopword: Option<String>,
    /// Notification channel for emitting agent events.
    pub notification_channel: Option<Sender<Notification>>,
    /// Optional reusable template for prompt building.
    pub template: Option<Arc<Mutex<Template>>>,
    /// Loaded Agent Skills available to this agent.
    pub skills: Vec<Skill>,
    /// Maximum allowed iterations during a conversation.
    pub max_iterations: Option<usize>,
    /// If true, clears history on every invocation.
    pub clear_history_on_invoke: bool,
    /// State for custom data
    pub state: HashMap<String, Value>,

    flow: Flow<I, O>,
    kind: PhantomData<(I, O)>,
}

impl<I, O> Agent<I, O> {
    pub(crate) async fn try_new(
        name: String,
        model: LlmModel,
        system_prompt: &str,
        local_tools: Option<Vec<Tool>>,
        response_format: Option<SchemaSpec>,
        stop_prompt: Option<String>,
        stopword: Option<String>,
        notification_channel: Option<Sender<Notification>>,
        provider_format: Option<Value>,
        mcp_servers: Option<Vec<McpServerType>>,
        flow: Flow<I, O>,
        template: Option<Arc<Mutex<Template>>>,
        skills: Vec<Skill>,
        max_iterations: Option<usize>,
        clear_history_on_invoke: bool,
    ) -> Result<Self, AgentBuildError> {
        let history = vec![Message::system(system_prompt.to_string())];

        let mut agent = Self {
            name,
            model,
            history,
            response_format,
            system_prompt: system_prompt.into(),
            stop_prompt,
            stopword,
            notification_channel,
            mcp_servers,
            local_tools,
            flow,
            tools: None,
            template,
            skills,
            max_iterations,
            clear_history_on_invoke,
            state: HashMap::new(),
            kind: PhantomData,
        };

        agent.tools = agent.get_compiled_tools().await?;
        agent
            .model
            .set_tools(agent.tools.clone())
            .set_response_format(agent.response_format.clone())
            .set_provider_format(provider_format)
            .set_name(Some(agent.name.clone()))
            .set_notification_channel(agent.notification_channel.clone());

        Ok(agent)
    }

    async fn invoke_text(&mut self, prompt: impl Into<String>) -> Result<Message, AgentError> {
        let prompt_str = prompt.into();

        let trace_span = span!(
            Level::INFO,
            "Invocation",
            "langfuse.observation.type" = "trace",
            "agent.model" = self.model.id(),
        );
        let parent_context = tracing::Span::current().context();
        let has_parent = parent_context.span().span_context().is_valid();
        if !has_parent {
            let name = self.name.clone();
            trace_span.set_attribute("langfuse.trace.name", name);
        }
        trace_span.set_attribute("langfuse.observation.input", prompt_str.clone());
        let _guard = trace_span.enter();

        let result = self
            .execute_invocation(prompt_str)
            .instrument(trace_span.clone())
            .await;

        match &result {
            Ok(message) => {
                if let Ok(json_output) = serde_json::to_string_pretty(message) {
                    trace_span.set_attribute("langfuse.observation.output", json_output);
                }
                trace_span.set_status(opentelemetry::trace::Status::Ok);
            }
            Err(e) => {
                trace_span.set_attribute("otel.status_code", "ERROR");
                trace_span.set_status(opentelemetry::trace::Status::Error {
                    description: e.to_string().into(),
                });
            }
        }

        result
    }

    async fn execute_invocation(&mut self, prompt: String) -> Result<Message, AgentError> {
        let flow_to_run = self.flow.clone();

        if self.clear_history_on_invoke {
            self.clear_history();
        }

        // // Record the specific prompt sent to the flow mechanism
        // Span::current().set_attribute("langfuse.observation.input", prompt.clone());

        let result = match flow_to_run {
            // These functions (invoke_nonstreaming/streaming) will create the "Generation" spans
            Flow::Default => default_flow(self, prompt).await,
            Flow::Func(custom_flow_fn) => (custom_flow_fn)(self, prompt).await,
        };

        // We can capture the raw output here as well for debugging the internal flow
        // if let Ok(msg) = &result {
        //     if let Some(content) = &msg.content {
        //         Span::current().set_attribute("langfuse.observation.output", content.clone());
        //     }
        // }

        result
    }

    /// Reset conversation history to contain only the system prompt.
    pub fn clear_history(&mut self) {
        self.history = vec![Message::system(self.system_prompt.clone())];
    }

    /// Persist the conversation history to disk in pretty-printed JSON.
    pub fn save_history<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let json_string = serde_json::to_string_pretty(&self.history)?;
        fs::write(path, json_string)?;
        Ok(())
    }

    /// Create a new notification channel for this agent.
    ///
    /// This re-initializes MCP tool connections so they bind to the new channel.
    pub async fn new_notification_channel(
        &mut self,
    ) -> Result<mpsc::Receiver<Notification>, AgentError> {
        let (s, r) = mpsc::channel::<Notification>(100);
        self.notification_channel = Some(s);
        self.tools = self.get_compiled_tools().await?;
        self.model
            .set_tools(self.tools.clone())
            .set_notification_channel(self.notification_channel.clone());
        Ok(r)
    }

    /// Build and return the tool set (local tools + MCP tools).
    pub async fn get_compiled_tools(&self) -> Result<Option<Vec<Tool>>, AgentBuildError> {
        let mut running_tools = self.local_tools.clone();

        match self.get_compiled_mcp_tools().await {
            Ok(tools_option) => {
                if let Some(mcp_tools) = tools_option {
                    match running_tools.as_mut() {
                        Some(t) => {
                            for mcpt in mcp_tools {
                                t.push(mcpt);
                            }
                        }
                        None => {
                            if !mcp_tools.is_empty() {
                                running_tools = Some(mcp_tools)
                            }
                        }
                    }
                }
            }
            Err(e) => return Err(e),
        }
        Ok(running_tools)
    }

    /// Build tool definitions from configured MCP servers.
    pub async fn get_compiled_mcp_tools(&self) -> Result<Option<Vec<Tool>>, AgentBuildError> {
        let mut running_tools: Option<Vec<Tool>> = None;
        if let Some(mcp_servers) = &self.mcp_servers {
            for mcp_server in mcp_servers {
                let mcp_tools = match get_mcp_tools(
                    mcp_server.clone(),
                    self.notification_channel.clone(),
                )
                .await
                {
                    Ok(t) => t,
                    Err(e) => return Err(AgentBuildError::McpError(e)),
                };

                match running_tools.as_mut() {
                    Some(t) => {
                        for mcpt in mcp_tools {
                            t.push(mcpt);
                        }
                    }
                    None => {
                        if !mcp_tools.is_empty() {
                            running_tools = Some(mcp_tools)
                        }
                    }
                }
            }
        }
        Ok(running_tools)
    }

    /// Find a tool reference by name, if it exists.
    pub fn get_tool_ref_by_name<T>(&self, name: T) -> Option<&Tool>
    where
        T: Into<String>,
    {
        let tools = self.tools.as_ref()?;

        let name = name.into();
        tools.iter().find(|&tool| tool.function.name.eq(&name))
    }

    /// Export current client configuration (provider, base URL, keys, etc.).
    pub fn export_client_config(&self) -> ClientConfig {
        self.model.client_config().clone()
    }

    /// Export current model configuration (temperature, top_p, penalties, etc.).
    pub fn export_model_config(&self) -> InferenceOptions {
        self.model.export_config()
    }

    /// Export prompt-level configuration (system prompt, tools, template, etc.).
    pub async fn export_prompt_config(&self) -> Result<PromptConfig, Error> {
        let template = if let Some(t) = self.template.clone() {
            Some(t.lock().await.clone())
        } else {
            None
        };

        let (response_format_raw, response_format) = if let Some(p) = self.response_format.clone() {
            (Some(serde_json::to_string(&p.schema)?), Some(p))
        } else {
            (None, None)
        };
        Ok(PromptConfig {
            template,
            system_prompt: Some(self.system_prompt.clone()),
            tools: self.tools.clone(),
            response_format,
            response_format_raw,
            mcp_servers: self.mcp_servers.clone(),
            stop_prompt: self.stop_prompt.clone(),
            stopword: self.stopword.clone(),
            max_iterations: self.max_iterations,
            clear_histroy_on_invoke: Some(self.clear_history_on_invoke),
            stream: self.model.stream_by_default(),
            pending_name: None,
            pending_strict: None,
        })
    }
}

impl Agent<Prompt, Standard> {
    pub async fn invoke(&mut self, prompt: impl Into<String>) -> Result<Message, AgentError> {
        self.invoke_text(prompt).await
    }
}

impl<T> Agent<Prompt, crate::Structured<T>>
where
    T: DeserializeOwned,
{
    pub async fn invoke(&mut self, prompt: impl Into<String>) -> Result<Message<T>, AgentError> {
        let message = self.invoke_text(prompt).await?;
        let id = message.id.clone();
        let encoded = serde_json::to_value(message)
            .map_err(|error| AgentError::Runtime(error.to_string()))?;
        let mut message: Message<T> =
            serde_json::from_value(encoded).map_err(AgentError::Deserialization)?;
        message.id = id;
        Ok(message)
    }
}

impl Agent<TemplateInput, Standard> {
    pub async fn invoke<K, V>(&mut self, data: HashMap<K, V>) -> Result<Message, AgentError>
    where
        K: Into<String> + Eq + std::hash::Hash,
        V: Into<String>,
    {
        let prompt = self.compile_template(data).await?;
        self.invoke_text(prompt).await
    }
}

impl<T> Agent<TemplateInput, crate::Structured<T>>
where
    T: DeserializeOwned,
{
    pub async fn invoke<K, V>(&mut self, data: HashMap<K, V>) -> Result<Message<T>, AgentError>
    where
        K: Into<String> + Eq + std::hash::Hash,
        V: Into<String>,
    {
        let prompt = self.compile_template(data).await?;
        let message = self.invoke_text(prompt).await?;
        let id = message.id.clone();
        let encoded = serde_json::to_value(message)
            .map_err(|error| AgentError::Runtime(error.to_string()))?;
        let mut message: Message<T> =
            serde_json::from_value(encoded).map_err(AgentError::Deserialization)?;
        message.id = id;
        Ok(message)
    }
}

impl<O> Agent<TemplateInput, O> {
    async fn compile_template<K, V>(&self, data: HashMap<K, V>) -> Result<String, AgentError>
    where
        K: Into<String> + Eq + std::hash::Hash,
        V: Into<String>,
    {
        let Some(template) = &self.template else {
            return Err(AgentError::Runtime("No template defined".into()));
        };
        let data = data
            .into_iter()
            .map(|(key, value)| (key.into(), value.into()))
            .collect::<HashMap<String, String>>();
        Ok(template.lock().await.compile(&data).await)
    }
}

impl<I, O> fmt::Debug for Agent<I, O> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Agent")
            .field("model", &self.model)
            .field("history", &self.history)
            .field("local_tools", &self.local_tools)
            .field("response_format", &self.response_format)
            .field("system_prompt", &self.system_prompt)
            .field("stop_prompt", &self.stop_prompt)
            .field("stopword", &self.stopword)
            .field("notification_channel", &self.notification_channel)
            .field("mcp_servers", &self.mcp_servers)
            .field("skills", &self.skills)
            .finish()
    }
}

impl<I, O> NotificationHandler for Agent<I, O> {
    fn get_outgoing_channel(&self) -> &Option<Sender<Notification>> {
        &self.notification_channel
    }

    fn get_channel_name(&self) -> &String {
        &self.name
    }
}
