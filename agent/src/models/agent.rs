use std::collections::HashMap;

use anyhow::Result;

use crate::services::ollama::{client::OllamaClient, models::{BaseRequest, ChatRequest, ChatResponse, Function, FunctionParameters, Message, Property, Tool, ToolType}};

use super::AgentError;


#[derive(Debug)]
pub struct Agent {
    model: String,
    history: Vec<Message>,
    ollama_client: OllamaClient,
    tools: Option<Vec<Tool>>,
}

impl Agent {
    pub(crate) fn new(
        model: &str,
        ollama_host: &str,
        ollama_port: u16,
        system_prompt: &str,
        tools: Option<Vec<Tool>>
    ) -> Self {
        let base_url = format!("{}:{}", ollama_host, ollama_port);
        let history = vec![Message::system(system_prompt.to_string())];

        Self {
            model: model.into(),
            history,
            ollama_client: OllamaClient::new(base_url),
            tools
        }
    }

    pub async fn invoke<T>(&mut self, prompt: T) -> Result<Message, AgentError>
    where
        T: Into<String>,
    {
        self.history.push(Message::user(prompt.into()));

        let request = ChatRequest {
            base: BaseRequest {
                model: self.model.clone(),
                format: None,
                options: None,
                stream: Some(false), 
                keep_alive: Some("5m".to_string()),
            },
            messages: self.history.clone(),
            tools: self.tools.clone(), 
        };

        let response: ChatResponse = self.ollama_client.chat(request).await?;
        self.history.push(response.message.clone());
        Ok(response.message)
    }
}