
use crate::services::{mcp::mcp_tool_builder::{get_mcp_tools, McpServerType}, ollama::models::tool::Tool};

use super::{Agent, AgentBuildError};

#[derive(Debug, Default)]
pub struct AgentBuilder {
    model: Option<String>,
    ollama_url: Option<String>,
    ollama_port: Option<u16>,
    system_prompt: Option<String>,
    tools: Option<Vec<Tool>>,
    response_format: Option<String>,
    mcp_server: Option<String>,
}


impl AgentBuilder { 
    pub fn set_model<T>(mut self, model: T) -> Self where T: Into<String> {
        self.model = Some(model.into());
        self
    }

    pub fn set_ollama_endpoint<T>(mut self, url: T) -> Self where T: Into<String> {
        self.ollama_url = Some(url.into());
        self
    }

    pub fn set_ollama_port(mut self, port: u16) -> Self {
        self.ollama_port = Some(port);
        self
    }

    pub fn set_system_prompt<T>(mut self, prompt: T) -> Self where T: Into<String> {
        self.system_prompt = Some(prompt.into());
        self
    }

    pub fn set_response_format<T>(mut self, format: T) -> Self where T: Into<String> {
        self.response_format = Some(format.into());
        self
    }
    
    pub fn add_tool(mut self, tool: Tool) -> Self {
        match self.tools.as_mut() {
            Some(vec_tools) => vec_tools.push(tool),
            None => self.tools = Some(vec![tool]),
        };
        self
    }

    pub fn add_mcp_server<T>(mut self, url: T) -> Self where T: Into<String> {
        self.mcp_server = Some(url.into());
        self
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
        
        if let Some(server_url) = self.mcp_server {
            let mcp_tools = match get_mcp_tools(server_url, McpServerType::Sse).await {
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

        Ok(Agent::new(
            &model, 
            &ollama_url, 
            ollama_port, 
            &system_prompt, 
            tools,
            response_format,
        ))
    }
}