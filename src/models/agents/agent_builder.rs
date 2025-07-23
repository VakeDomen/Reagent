use std::sync::Arc;

use tokio::sync::{mpsc, Mutex};
use crate::{
    models::{
        agents::flow::invocation_flows::Flow,
        notification::Notification, AgentBuildError
    },
    util::templating::Template,
    services::{
        mcp::mcp_tool_builder::McpServerType,
        ollama::models::tool::Tool
    }, Agent
};

/// A builder for [`Agent`].
///
/// Allows configuration of model, endpoint, tools, penalties, flow, etc.
/// All methods take `self` and return `Self`, so you can chain calls.
///
/// # Examples
///
/// ```rust
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// use reagent::AgentBuilder;
/// let agent = AgentBuilder::default()
///     .set_model("mistral-nemo")
///     .set_ollama_port(8000)
///     .set_system_prompt("Be concise")
///     .strip_thinking(false)
///     .build()
///     .await?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Default)]
pub struct AgentBuilder {
    name: Option<String>,
    model: Option<String>,
    ollama_url: Option<String>,
    system_prompt: Option<String>,
    tools: Option<Vec<Tool>>,
    response_format: Option<String>,
    mcp_servers: Option<Vec<McpServerType>>,
    stop_prompt: Option<String>,
    stopword: Option<String>,
    strip_thinking: Option<bool>,
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
    top_k: Option<u32>,
    min_p: Option<f32>,
    notification_channel: Option<mpsc::Sender<Notification>>,
    flow: Option<Flow>,
    template: Option<Arc<Mutex<Template>>>,
}

impl AgentBuilder {

    /// Set the name of the agent (used in logging)
    pub fn set_name<T>(mut self, name: T) -> Self where T: Into<String> {
        self.name = Some(name.into());
        self
    }

    /// Set the sampling temperature.
    pub fn set_temperature(mut self, v: f32) -> Self {
        self.temperature = Some(v);
        self
    }

    /// Set nucleus sampling probability.
    pub fn set_top_p(mut self, v: f32) -> Self {
        self.top_p = Some(v);
        self
    }

    /// Set presence penalty.
    pub fn set_presence_penalty(mut self, v: f32) -> Self {
        self.presence_penalty = Some(v);
        self
    }

    /// Set frequency penalty.
    pub fn set_frequency_penalty(mut self, v: f32) -> Self {
        self.frequency_penalty = Some(v);
        self
    }

    /// Set maximum context length (in tokens/chunks).
    pub fn set_num_ctx(mut self, v: u32) -> Self {
        self.num_ctx = Some(v);
        self
    }

    /// Repeat penalty for the last N tokens.
    pub fn set_repeat_last_n(mut self, v: i32) -> Self {
        self.repeat_last_n = Some(v);
        self
    }

    /// Set penalty for repeated tokens.
    pub fn set_repeat_penalty(mut self, v: f32) -> Self {
        self.repeat_penalty = Some(v);
        self
    }

    /// Set RNG seed for sampling.
    pub fn set_seed(mut self, v: i32) -> Self {
        self.seed = Some(v);
        self
    }

    /// Set the hard stop string.
    pub fn set_stop<T: Into<String>>(mut self, v: T) -> Self {
        self.stop = Some(v.into());
        self
    }

    /// Number of tokens to predict.
    pub fn set_num_predict(mut self, v: i32) -> Self {
        self.num_predict = Some(v);
        self
    }

    /// Top-K sampling.
    pub fn set_top_k(mut self, v: u32) -> Self {
        self.top_k = Some(v);
        self
    }

    /// Minimum probability threshold.
    pub fn set_min_p(mut self, v: f32) -> Self {
        self.min_p = Some(v);
        self
    }

    /// Select the underlying model name.
    pub fn set_model<T: Into<String>>(mut self, model: T) -> Self {
        self.model = Some(model.into());
        self
    }

    /// URL of the Ollama service. Note port is set separately in `set_ollama_port`
    pub fn set_ollama_endpoint<T: Into<String>>(mut self, url: T) -> Self {
        self.ollama_url = Some(url.into());
        self
    }

    /// System prompt that initializes conversation history.
    pub fn set_system_prompt<T: Into<String>>(mut self, prompt: T) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// JSON schema string to constrain response format.
    pub fn set_response_format<T: Into<String>>(mut self, format: T) -> Self {
        self.response_format = Some(format.into());
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

    /// Choose the high‐level control flow policy.
    pub fn set_flow(mut self, flow: Flow) -> Self {
        self.flow = Some(flow);
        self
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

    /// Add an MCP server endpoint.
    pub fn add_mcp_server(mut self, server: McpServerType) -> Self {
        if let Some(ref mut svs) = self.mcp_servers {
            svs.push(server);
        } else {
            self.mcp_servers = Some(vec![server]);
        }
        self
    }

    /// Set a template for the agent's first prompt
    pub fn set_template(mut self, template: Template) -> Self {
        self.template = Some(Arc::new(Mutex::new(template)));
        self
    }

    /// Build an [`Agent`] and return also the notification receiver.
    ///
    /// Creates an internal mpsc channel of size 100.
    pub async fn build_with_notification(
        mut self
    ) -> Result<(Agent, mpsc::Receiver<Notification>), AgentBuildError> {
        let (sender, receiver) = mpsc::channel(100);
        self.notification_channel = Some(sender);
        let agent = self.build().await?;
        Ok((agent, receiver))
    }

    /// Finalize all settings and produce an [`Agent`], or an error if required fields missing or invalid.
    pub async fn build(self) -> Result<Agent, AgentBuildError> {
        let model = self.model.ok_or(AgentBuildError::ModelNotSet)?;
        let ollama_url = self.ollama_url.unwrap_or_else(|| "http://localhost:11434".into());
        let system_prompt = self.system_prompt.unwrap_or_else(|| "You are a helpful agent.".into());
        let strip_thinking = self.strip_thinking.unwrap_or(true);

        let response_format = if let Some(schema) = self.response_format {
            let trimmed = schema.trim();
            match serde_json::from_str(trimmed) {
                Ok(v) => Some(v),
                Err(e) => {
                    return Err(AgentBuildError::InvalidJsonSchema(format!(
                        "Failed to parse JSON schema `{trimmed}`: {e}"
                    )))
                }
            }
        } else {
            None
        };

        let flow = self.flow.unwrap_or(Flow::Default);

        let name = match self.name {
            Some(n) => n,
            None => format!("Agent-{}", model),
        };

        Ok(Agent::new(
            name,
            &model,
            &ollama_url,
            &system_prompt,
            self.tools.clone(),
            response_format,
            self.stop_prompt,
            self.stopword,
            strip_thinking,
            self.temperature,
            self.top_p,
            self.presence_penalty,
            self.frequency_penalty,
            self.num_ctx,
            self.repeat_last_n,
            self.repeat_penalty,
            self.seed,
            self.stop,
            self.num_predict,
            self.top_k,
            self.min_p,
            self.notification_channel,
            self.mcp_servers,
            flow.into(),
            self.template,
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use serde_json::Value;

    use super::*;
    use crate::{models::agents::flow::invocation_flows::{Flow, FlowFuture}, Agent, AsyncToolFn, Message, ToolBuilder};

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
            .set_response_format(json)
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
            .set_response_format(bad)
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
                    "#.into())
                })
            })
        };

        let weather_tool = ToolBuilder::new()
            .function_name("get_current_weather")
            .function_description("Returns a weather forecast for a given location")
            .add_property("location", "string", "City name")
            .add_required_property("location")
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
            .send(Notification::Done(false))
            .await
            .unwrap();
        let notified = rx.recv().await.unwrap();
        assert!(matches!(notified, Notification::Done(false)));
    }

    #[tokio::test]
    async fn custom_flow_invocation() {
        
        fn echo_flow<'a>(_agent: &'a mut Agent, prompt: String) -> FlowFuture<'a> {
            Box::pin(async move {
                    Ok(Message::system(format!("ECHO: {}", prompt)))
            })    
        }

        let agent = AgentBuilder::default()
            .set_model("m")
            .set_flow(Flow::Custom(echo_flow))
            .build()
            .await
            .unwrap();
        let mut a = agent.clone();
        let resp = a.invoke_flow("abc").await.unwrap();
        assert_eq!(resp.content.unwrap(), "ECHO: abc");
    }
}
