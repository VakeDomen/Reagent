use anyhow::Result;


use crate::services::ollama::{client::OllamaClient, models::{base::{BaseRequest, Message}, chat::{ChatRequest, ChatResponse}, tool::{Tool, ToolCall}}};

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
        let message = response.message.clone();
        
        if let Some(tc) = &message.tool_calls {
            self.call_tools(tc).await
        }
        
        self.history.push(message);
        Ok(response.message)
    }


    async fn call_tools(&self, tool_calls: &Vec<ToolCall>) {
        if let Some(avalible_tools) = &self.tools {
            for tool_call in tool_calls {
                for avalible_tool in avalible_tools {
                    if avalible_tool.function.name.eq(&tool_call.function.name) {
                        let _ = avalible_tool.execute(tool_call.function.arguments.clone()).await;
                    }
                }
            }
        }
    }
}