
use crate::{models::AgentError, services::ollama::models::{base::{BaseRequest, OllamaOptions}, chat::ChatRequest, tool::Tool}, Agent, Message};





/// Common parameters for building a [`ChatRequest`].
pub struct RequestParams {
    pub model:       String,
    pub format:      Option<serde_json::Value>,
    pub options:     OllamaOptions,
    pub stream:      bool,
    pub keep_alive:  String,
    pub messages:    Vec<Message>,
    pub tools:       Option<Vec<Tool>>,
}

impl RequestParams {
    /// Turn these params into a [`ChatRequest`].
    pub fn into_request(self) -> ChatRequest {
        ChatRequest {
            base: BaseRequest {
                model:      self.model,
                format:     self.format,
                options:    Some(self.options),
                stream:     Some(self.stream),
                keep_alive: Some(self.keep_alive),
            },
            messages: self.messages,
            tools:    self.tools,
        }
    }
}

/// Convert something into the shared [`RequestParams`].
pub trait ToRequestParams {
    fn to_request_params(&self) -> RequestParams;
}

impl ToRequestParams for Agent {
    fn to_request_params(&self) -> RequestParams {
        RequestParams {
            model:      self.model.clone(),
            format:     self.response_format.clone(),
            options:    OllamaOptions {
                num_ctx:            self.num_ctx,
                repeat_last_n:      self.repeat_last_n,
                repeat_penalty:     self.repeat_penalty,
                temperature:        self.temperature,
                seed:               self.seed,
                stop:               self.stop.clone(),
                num_predict:        self.num_predict,
                top_k:              self.top_k,
                top_p:              self.top_p,
                min_p:              self.min_p,
                presence_penalty:   self.presence_penalty,
                frequency_penalty:  self.frequency_penalty,
            },
            stream:     false,
            keep_alive: "5m".to_string(),
            messages:   self.history.clone(),
            tools:      self.tools.clone(),
        }
    }
}


/// Builds a [`ChatRequest`] from the agent’s state, *including* whatever
/// `agent.tools` currently holds.  
///
/// # Errors  
pub async fn generate_llm_request(
    agent: &Agent
) -> ChatRequest {
    let params = agent.to_request_params();
    params.into_request()
}


/// Like [`generate_llm_request`] but always sets `tools: None` in the request.
pub async fn generate_llm_request_without_tools(
    agent: &Agent
) -> ChatRequest {
    let mut params = agent.to_request_params();
    params.tools = None;
    params.into_request()
}



/// Like [`generate_llm_request`] but uses a custom message list
/// instead of the agent's history. Intended for ReAct-style flows.
pub fn generate_custom_request(
    agent: &Agent,
    messages: Vec<Message>,
    tools: Option<Vec<Tool>>,
) -> ChatRequest {
    let mut params = agent.to_request_params();
    params.messages = messages;
    params.tools    = tools;
    params.into_request()
}
