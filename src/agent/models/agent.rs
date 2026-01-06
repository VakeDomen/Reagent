use crate::agent::models::configs::{ModelConfig, PromptConfig};
use crate::agent::models::error::{AgentBuildError, AgentError};
use crate::services::llm::{ClientConfig, InferenceClient, SchemaSpec};
use crate::templates::Template;
use crate::{default_flow, Flow, NotificationHandler};
use core::fmt;
use opentelemetry::trace::TraceContextExt;
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::{Error, Value};
use std::sync::Arc;
use std::{collections::HashMap, fs, path::Path};
use tokio::sync::mpsc::{self, Sender};
use tokio::sync::Mutex;
use tracing::{instrument, span, Instrument, Level, Span};
use tracing_opentelemetry::OpenTelemetrySpanExt;

use crate::{
    notifications::Notification,
    services::{llm::models::message::Message, mcp::mcp_tool_builder::get_mcp_tools},
    McpServerType, Tool,
};

#[derive(Clone)]
pub struct Agent {
    /// Human-readable name of the agent.
    pub name: String,
    /// Underlying model identifier.
    pub model: String,
    /// Conversation history with the model.
    pub history: Vec<Message>,
    /// Locally registered tools (before MCP merge).
    pub local_tools: Option<Vec<Tool>>,
    /// Configured MCP server endpoints.
    pub mcp_servers: Option<Vec<McpServerType>>,
    /// Fully compiled tool set (local + MCP).
    pub tools: Option<Vec<Tool>>,
    /// JSON schema format for responses, if any.
    pub response_format: Option<Value>,
    /// Backend model client.
    pub(crate) inference_client: InferenceClient,
    /// System prompt injected at the start of the conversation.
    pub system_prompt: String,
    /// Optional stop prompt inserted on tool branches.
    pub stop_prompt: Option<String>,
    /// Stopword to detect end of generation.
    pub stopword: Option<String>,
    /// Whether `<think>` blocks should be stripped from outputs.
    pub strip_thinking: bool,
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Nucleus sampling top-p parameter.
    pub top_p: Option<f32>,
    /// Presence penalty parameter.
    pub presence_penalty: Option<f32>,
    /// Frequency penalty parameter.
    pub frequency_penalty: Option<f32>,
    /// Maximum context window size.
    pub num_ctx: Option<u32>,
    /// Last-N window for repetition penalty.
    pub repeat_last_n: Option<i32>,
    /// Repetition penalty multiplier.
    pub repeat_penalty: Option<f32>,
    /// RNG seed for reproducibility.
    pub seed: Option<i32>,
    /// Hard stop sequence.
    pub stop: Option<String>,
    /// Maximum tokens to predict.
    pub num_predict: Option<i32>,
    /// Top-K sampling cutoff.
    pub top_k: Option<u32>,
    /// Minimum probability threshold.
    pub min_p: Option<f32>,
    /// Keep alive - keep model in memory
    pub keep_alive: Option<String>,
    /// Whether to stream token notifications.
    pub stream: bool,
    /// Notification channel for emitting agent events.
    pub notification_channel: Option<Sender<Notification>>,
    /// Optional reusable template for prompt building.
    pub template: Option<Arc<Mutex<Template>>>,
    /// Maximum allowed iterations during a conversation.
    pub max_iterations: Option<usize>,
    /// If true, clears history on every invocation.
    pub clear_history_on_invoke: bool,
    /// State for custom data
    pub state: HashMap<String, Value>,

    flow: Flow,
}

impl Agent {
    pub(crate) async fn try_new(
        name: String,
        model: &str,
        inference_client: InferenceClient,
        system_prompt: &str,
        local_tools: Option<Vec<Tool>>,
        response_format: Option<Value>,
        stop_prompt: Option<String>,
        stopword: Option<String>,
        strip_thinking: bool,
        temperature: Option<f32>,
        top_p: Option<f32>,
        presence_penalty: Option<f32>,
        frequency_penalty: Option<f32>,
        num_ctx: Option<u32>,
        repeat_last_n: Option<i32>,
        repeat_penalty: Option<f32>,
        seed: Option<i32>,
        stop: Option<String>,
        num_predict: Option<i32>,
        stream: bool,
        top_k: Option<u32>,
        min_p: Option<f32>,
        keep_alive: Option<String>,
        notification_channel: Option<Sender<Notification>>,
        mcp_servers: Option<Vec<McpServerType>>,
        flow: Flow,
        template: Option<Arc<Mutex<Template>>>,
        max_iterations: Option<usize>,
        clear_history_on_invoke: bool,
    ) -> Result<Self, AgentBuildError> {
        let history = vec![Message::system(system_prompt.to_string())];

        let mut agent = Self {
            name,
            model: model.into(),
            history,
            inference_client,
            response_format,
            system_prompt: system_prompt.into(),
            stop_prompt,
            stopword,
            strip_thinking,
            temperature,
            top_p,
            presence_penalty,
            frequency_penalty,
            num_ctx,
            repeat_last_n,
            repeat_penalty,
            seed,
            stop,
            num_predict,
            top_k,
            min_p,
            keep_alive,
            notification_channel,
            mcp_servers,
            local_tools,
            flow,
            tools: None,
            template,
            max_iterations,
            clear_history_on_invoke,
            stream,
            state: HashMap::new(),
        };

        agent.tools = agent.get_compiled_tools().await?;

        Ok(agent)
    }

    /// Invoke the agent with a raw string prompt.
    ///
    /// This is the most direct way to ask the agent something:
    /// the given prompt string it is conveterd to a user message and
    /// appended to history. It is passed through
    /// the configured [`Flow`] (either `Default` or `Custom`).
    ///
    /// Returns the raw [`Message`] produced by the flow.
    pub async fn invoke_flow<T>(&mut self, prompt: T) -> Result<Message, AgentError>
    where
        T: Into<String>,
    {
        let prompt_str = prompt.into();

        let trace_span = span!(
            Level::INFO,
            "Invocation",
            "langfuse.observation.type" = "trace",
            "agent.model" = self.model.as_str(),
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

    /// Invoke the agent expecting structured JSON output.
    ///
    /// Works like [`invoke_flow`], but attempts to deserialize the
    /// model’s response into type `O` which must be deserializable.
    ///
    /// Use this when you constrain the response with a JSON schema
    /// (`response_format`) and want the result to be typed.
    pub async fn invoke_flow_structured_output<T, O>(&mut self, prompt: T) -> Result<O, AgentError>
    where
        T: Into<String>,
        O: DeserializeOwned + Serialize,
    {
        let prompt_str = prompt.into();

        let trace_span = span!(
            Level::INFO,
            "Invocation with structured output",
            "langfuse.observation.type" = "trace",
            "agent.model" = self.model.as_str(),
        );
        let parent_context = tracing::Span::current().context();
        let has_parent = parent_context.span().span_context().is_valid();
        if !has_parent {
            let name = self.name.clone();
            trace_span.set_attribute("langfuse.trace.name", name);
        }
        let _guard = trace_span.enter();

        trace_span.set_attribute("langfuse.observation.input", prompt_str.clone());

        // Logic (inlined slightly to capture intermediate steps if needed,
        // but calling execute_invocation is cleaner)
        let response_result = self.execute_invocation(prompt_str).await;

        match response_result {
            Ok(response) => {
                let Some(json) = response.content else {
                    let e = AgentError::Runtime("Agent did not produce content".into());
                    trace_span.set_status(opentelemetry::trace::Status::Error {
                        description: e.to_string().into(),
                    });
                    return Err(e);
                };

                // Deserialize to O
                match serde_json::from_str::<O>(&json).map_err(AgentError::Deserialization) {
                    Ok(out) => {
                        // Serialize O back to string to record it as the Trace Output
                        if let Ok(dump) = serde_json::to_string_pretty(&out) {
                            trace_span.set_attribute("langfuse.observation.output", dump);
                        }
                        trace_span.set_status(opentelemetry::trace::Status::Ok);
                        Ok(out)
                    }
                    Err(e) => {
                        trace_span.set_status(opentelemetry::trace::Status::Error {
                            description: e.to_string().into(),
                        });
                        Err(e)
                    }
                }
            }
            Err(e) => {
                trace_span.set_status(opentelemetry::trace::Status::Error {
                    description: e.to_string().into(),
                });
                Err(e)
            }
        }
    }

    /// Invoke the agent using a prompt compiled from a template.
    ///
    /// The provided `template_data` is substituted into the configured
    /// [`Template`] before invoking the flow. This allows building prompts
    /// from reusable templates instead of raw strings.
    ///
    /// Returns the raw [`Message`] produced by the flow.
    pub async fn invoke_flow_with_template<K, V>(
        &mut self,
        template_data: HashMap<K, V>,
    ) -> Result<Message, AgentError>
    where
        K: Into<String> + serde::Serialize, // Added Serialize for trace input
        V: Into<String> + serde::Serialize,
    {
        // We need to convert the generic HashMap to the specific map required by the template engine
        // AND keep a copy for the Trace input.

        // 1. Prepare data for Tracing
        let trace_input = serde_json::to_string_pretty(&template_data).unwrap_or_default();

        // 2. Prepare data for Compilation
        let string_map: HashMap<String, String> = template_data
            .into_iter()
            .map(|(k, v)| (k.into(), v.into()))
            .collect();

        let trace_span = span!(
            Level::INFO,
            "Invocation with template",
            "langfuse.observation.type" = "trace",
            "agent.model" = self.model.as_str(),
        );
        let parent_context = tracing::Span::current().context();
        let has_parent = parent_context.span().span_context().is_valid();
        if !has_parent {
            let name = self.name.clone();
            trace_span.set_attribute("langfuse.trace.name", name);
        }
        trace_span.set_attribute("langfuse.observation.input", trace_input);
        let _guard = trace_span.enter();

        let Some(template) = &self.template else {
            let e = AgentError::Runtime("No template defined".into());
            trace_span.set_status(opentelemetry::trace::Status::Error {
                description: e.to_string().into(),
            });
            return Err(e);
        };

        // Compile prompt (This could be its own span if compilation is complex)
        let prompt = { template.lock().await.compile(&string_map).await };

        // Execute
        let result = self.execute_invocation(prompt).await;

        match &result {
            Ok(msg) => {
                if let Ok(out) = serde_json::to_string_pretty(msg) {
                    trace_span.set_attribute("langfuse.observation.output", out);
                }
                trace_span.set_status(opentelemetry::trace::Status::Ok);
            }
            Err(e) => {
                trace_span.set_status(opentelemetry::trace::Status::Error {
                    description: e.to_string().into(),
                });
            }
        }

        result
    }

    /// Invoke the agent with a template and parse structured output.
    ///
    /// Combines [`invoke_flow_with_template`] with [`invoke_flow_structured_output`]:
    /// first compiles the prompt from the agent’s [`Template`] and `template_data`,
    /// then invokes the flow and tries to deserialize the result into type `O`.
    ///
    /// Use this when you constrain the response with a JSON schema
    /// (`response_format`) and want the result to be typed.
    pub async fn invoke_flow_with_template_structured_output<K, V, O>(
        &mut self,
        template_data: HashMap<K, V>,
    ) -> Result<O, AgentError>
    where
        K: Into<String> + serde::Serialize,
        V: Into<String> + serde::Serialize,
        O: DeserializeOwned + serde::Serialize,
    {
        let trace_input = serde_json::to_string_pretty(&template_data).unwrap_or_default();

        let string_map: HashMap<String, String> = template_data
            .into_iter()
            .map(|(k, v)| (k.into(), v.into()))
            .collect();

        let trace_span = span!(
            Level::INFO,
            "Invocation with template and structured output",
            "langfuse.observation.type" = "trace",
            "agent.model" = self.model.as_str(),
        );
        let parent_context = tracing::Span::current().context();
        let has_parent = parent_context.span().span_context().is_valid();
        if !has_parent {
            let name = self.name.clone();
            trace_span.set_attribute("langfuse.trace.name", name);
        }
        trace_span.set_attribute("langfuse.observation.input", trace_input);
        let _guard = trace_span.enter();

        let Some(template) = &self.template else {
            return Err(AgentError::Runtime("No template defined".into()));
        };

        let prompt = { template.lock().await.compile(&string_map).await };

        let response_result = self.execute_invocation(prompt).await;

        match response_result {
            Ok(response) => {
                let Some(json) = response.content else {
                    let e = AgentError::Runtime("Agent did not produce content".into());
                    trace_span.set_status(opentelemetry::trace::Status::Error {
                        description: e.to_string().into(),
                    });
                    return Err(e);
                };
                match serde_json::from_str::<O>(&json).map_err(AgentError::Deserialization) {
                    Ok(out) => {
                        if let Ok(dump) = serde_json::to_string_pretty(&out) {
                            trace_span.set_attribute("langfuse.observation.output", dump);
                        }
                        trace_span.set_status(opentelemetry::trace::Status::Ok);
                        Ok(out)
                    }
                    Err(e) => {
                        trace_span.set_status(opentelemetry::trace::Status::Error {
                            description: e.to_string().into(),
                        });
                        Err(e)
                    }
                }
            }
            Err(e) => {
                trace_span.set_status(opentelemetry::trace::Status::Error {
                    description: e.to_string().into(),
                });
                Err(e)
            }
        }
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
        self.inference_client.get_config().clone()
    }

    /// Export current model configuration (temperature, top_p, penalties, etc.).
    pub fn export_model_config(&self) -> ModelConfig {
        ModelConfig {
            model: Some(self.model.clone()),
            temperature: self.temperature,
            top_p: self.top_p,
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            num_ctx: self.num_ctx,
            repeat_last_n: self.repeat_last_n,
            repeat_penalty: self.repeat_penalty,
            seed: self.seed,
            stop: self.stop.clone(),
            num_predict: self.num_predict,
            top_k: self.top_k,
            min_p: self.min_p,
        }
    }

    /// Export prompt-level configuration (system prompt, tools, template, etc.).
    pub async fn export_prompt_config(&self) -> Result<PromptConfig, Error> {
        let template = if let Some(t) = self.template.clone() {
            Some(t.lock().await.clone())
        } else {
            None
        };

        let (response_format_raw, response_format) = if let Some(p) = self.response_format.clone() {
            (
                Some(serde_json::to_string(&p)?),
                Some(SchemaSpec::from_value(p)),
            )
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
            strip_thinking: Some(self.strip_thinking),
            max_iterations: self.max_iterations,
            clear_histroy_on_invoke: Some(self.clear_history_on_invoke),
            stream: self.stream,
            pending_name: None,
            pending_strict: None,
        })
    }
}

impl fmt::Debug for Agent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Agent")
            .field("model", &self.model)
            .field("history", &self.history)
            .field("local_tools", &self.local_tools)
            .field("response_format", &self.response_format)
            .field("inference_client", &self.inference_client)
            .field("system_prompt", &self.system_prompt)
            .field("stop_prompt", &self.stop_prompt)
            .field("stopword", &self.stopword)
            .field("strip_thinking", &self.strip_thinking)
            .field("temperature", &self.temperature)
            .field("top_p", &self.top_p)
            .field("presence_penalty", &self.presence_penalty)
            .field("frequency_penalty", &self.frequency_penalty)
            .field("num_ctx", &self.num_ctx)
            .field("repeat_last_n", &self.repeat_last_n)
            .field("repeat_penalty", &self.repeat_penalty)
            .field("seed", &self.seed)
            .field("stop", &self.stop)
            .field("num_predict", &self.num_predict)
            .field("top_k", &self.top_k)
            .field("min_p", &self.min_p)
            .field("notification_channel", &self.notification_channel)
            .field("mcp_servers", &self.mcp_servers)
            .finish()
    }
}

impl NotificationHandler for Agent {
    fn get_outgoing_channel(&self) -> &Option<Sender<Notification>> {
        &self.notification_channel
    }

    fn get_channel_name(&self) -> &String {
        &self.name
    }
}
