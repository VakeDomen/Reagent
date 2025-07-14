
use tokio::{sync::mpsc};

use crate::{models::notification::Notification, services::{mcp::mcp_tool_builder::{get_mcp_tools, McpServerType}, ollama::models::tool::Tool}};

use super::{Agent, AgentBuildError};

#[derive(Debug, Default)]
pub struct AgentBuilder {
    model: Option<String>,
    ollama_url: Option<String>,
    ollama_port: Option<u16>,
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
    notification_channel: Option<mpsc::Sender<Notification>>
}


impl AgentBuilder { 

    pub fn set_temperature(mut self, v: f32) -> Self { self.temperature = Some(v); self }
    pub fn set_top_p(mut self, v: f32) -> Self { self.top_p = Some(v); self }
    pub fn set_presence_penalty(mut self, v: f32) -> Self { self.presence_penalty = Some(v); self }
    pub fn set_frequency_penalty(mut self, v: f32) -> Self { self.frequency_penalty = Some(v); self }
    pub fn set_num_ctx(mut self, v: u32) -> Self { self.num_ctx = Some(v); self }
    pub fn set_repeat_last_n(mut self, v: i32) -> Self { self.repeat_last_n = Some(v); self }
    pub fn set_repeat_penalty(mut self, v: f32) -> Self { self.repeat_penalty = Some(v); self }
    pub fn set_seed(mut self, v: i32) -> Self { self.seed = Some(v); self }
    pub fn set_stop<T: Into<String>>(mut self, v: T) -> Self { self.stop = Some(v.into()); self }
    pub fn set_num_predict(mut self, v: i32) -> Self { self.num_predict = Some(v); self }
    pub fn set_top_k(mut self, v: u32) -> Self { self.top_k = Some(v); self }
    pub fn set_min_p(mut self, v: f32) -> Self { self.min_p = Some(v); self }
    pub fn set_model<T>(mut self, model: T) -> Self where T: Into<String> { self.model = Some(model.into()); self }
    pub fn set_ollama_endpoint<T>(mut self, url: T) -> Self where T: Into<String> { self.ollama_url = Some(url.into()); self }
    pub fn set_ollama_port(mut self, port: u16) -> Self { self.ollama_port = Some(port); self }
    pub fn set_system_prompt<T>(mut self, prompt: T) -> Self where T: Into<String> { self.system_prompt = Some(prompt.into()); self }
    pub fn set_response_format<T>(mut self, format: T) -> Self where T: Into<String> { self.response_format = Some(format.into()); self }
    pub fn set_stop_prompt<T>(mut self, stop_prompt: T) -> Self where T: Into<String> { self.stop_prompt = Some(stop_prompt.into()); self }
    pub fn set_stopword<T>(mut self, stopword: T) -> Self where T: Into<String> { self.stopword = Some(stopword.into()); self }
    pub fn strip_thinking(mut self, strip: bool) -> Self { self.strip_thinking = Some(strip); self }
    
    pub fn add_tool(mut self, tool: Tool) -> Self {
        match self.tools.as_mut() {
            Some(vec_tools) => vec_tools.push(tool),
            None => self.tools = Some(vec![tool]),
        };
        self
    }

    pub fn add_mcp_server(mut self, server: McpServerType) -> Self {
        match self.mcp_servers.as_mut() {
            Some(servers) => servers.push(server),
            None => self.mcp_servers = Some(vec![server]),
        }
        self
    }

    pub async fn build_with_notification(mut self) -> Result<(Agent, mpsc::Receiver<Notification>), AgentBuildError> {
        let (s, r) = mpsc::channel::<Notification>(100);
        self.notification_channel = Some(s);
        let agent = self.build().await?;
        Ok((agent, r))
    }

    pub async fn build(self) -> Result<Agent, AgentBuildError> {
        let model = match self.model {
            Some(m) => m,
            None => return Err(AgentBuildError::ModelNotSet),
        };
        
        let ollama_url = match self.ollama_url {
            Some(m) => m,
            None => "http://localhost".into(),
        };
        
        let ollama_port = match self.ollama_port {
            Some(m) => m,
            None => 11434,
        };
        
        let system_prompt = match self.system_prompt {
            Some(m) => m,
            None => "You are a helpful agent.".into(),
        };

        let strip_thinking = match self.strip_thinking {
            Some(s) => s,
            None => true,
        };
        
        let mut response_format = None;
        if let Some(schema_str) = self.response_format {
            let trimmed_schema_str = schema_str.trim();
            match serde_json::from_str(trimmed_schema_str) {
                Ok(parsed_schema_object) => response_format = Some(parsed_schema_object),
                Err(e) => return Err(AgentBuildError::InvalidJsonSchema(format!(
                        "Failed to parse provided JSON schema string: {}. Error: {}",
                        trimmed_schema_str, e
                )))
            }
        }

        let mut tools = self.tools.clone();
        
        if let Some(mcp_servers) = self.mcp_servers {
            for mcp_server in mcp_servers {
                let mcp_tools = match get_mcp_tools(mcp_server, self.notification_channel.clone()).await {
                    Ok(t) => t,
                    Err(e) => return Err(AgentBuildError::McpError(e)),
                };
    
                match tools.as_mut() {
                    Some(t) => for mcpt in mcp_tools { t.push(mcpt); },
                    None => if mcp_tools.len() > 0 {
                        tools = Some(mcp_tools)
                    },
                }
            }

        }

        Ok(Agent::new(
            &model, 
            &ollama_url, 
            ollama_port, 
            &system_prompt, 
            tools,
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
        ))
    }
}


#[cfg(test)]
mod tests {
    use super::*; 

    #[tokio::test]
    async fn agent_builder_defaults() {
        // Assuming ModelNotSet is an error if set_model is not called
        let builder_result = AgentBuilder::default().build().await;
        assert!(builder_result.is_err());
        match builder_result.unwrap_err() {
            AgentBuildError::ModelNotSet => {} // Expected
            _ => panic!("Expected ModelNotSet error"),
        }

        let agent = AgentBuilder::default()
            .set_model("test-model")
            .build()
            .await
            .expect("Agent build failed with default settings");

        assert_eq!(agent.model, "test-model"); // Add a getter for model name in Agent
        assert!(agent.response_format.is_none()); // Add getter
        assert!(agent.tools.is_none() || agent.tools.unwrap().is_empty()); // Add getter
    }

    #[tokio::test]
    async fn agent_builder_custom_settings() {
        let agent = AgentBuilder::default()
            .set_model("custom-model")
            .set_ollama_endpoint("http://custom-ollama")
            .set_ollama_port(12345)
            .set_system_prompt("Custom prompt")
            .set_response_format(r#"{"type": "object", "properties": {"key": {"type": "string"}}}"#)
            .build()
            .await
            .expect("Agent build failed with custom settings");

        assert_eq!(agent.model, "custom-model");
        assert_eq!(agent.history.first().unwrap().content.clone().unwrap(), "Custom prompt"); // Add getter
        assert!(agent.response_format.is_some());
        assert_eq!(agent.response_format.unwrap().get("type").unwrap().as_str().unwrap(), "object");
    }

}
