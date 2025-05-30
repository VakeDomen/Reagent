use serde_json::Value;

use crate::services::ollama::{client::OllamaClient, models::{base::{BaseRequest, Message}, chat::{ChatRequest, ChatResponse}, tool::{Tool, ToolCall}}};

use super::AgentError;


#[derive(Debug)]
pub struct Agent {
    pub model: String,
    pub history: Vec<Message>,
    pub tools: Option<Vec<Tool>>,
    pub response_format: Option<Value>,
    ollama_client: OllamaClient,
}

impl Agent {
    pub(crate) fn new(
        model: &str,
        ollama_host: &str,
        ollama_port: u16,
        system_prompt: &str,
        tools: Option<Vec<Tool>>,
        response_format: Option<Value>,
    ) -> Self {
        let base_url = format!("{}:{}", ollama_host, ollama_port);
        let history = vec![Message::system(system_prompt.to_string())];

        Self {
            model: model.into(),
            history,
            ollama_client: OllamaClient::new(base_url),
            tools,
            response_format
        }
    }

    pub async fn invoke<T>(&mut self, prompt: T) -> Result<Message, AgentError>
    where
        T: Into<String>,
    {
        self.history.push(Message::user(prompt.into()));

        loop {
            
            let request = ChatRequest {
                base: BaseRequest {
                    model: self.model.clone(),
                    format: self.response_format.clone(),
                    options: None,
                    stream: Some(false), 
                    keep_alive: Some("5m".to_string()),
                },
                messages: self.history.clone(),
                tools: self.tools.clone(), 
            };
    
            let response: ChatResponse = self.ollama_client.chat(request).await?;
            let message = response.message.clone();
           
            let tool_calls = message.tool_calls.clone();
            self.history.push(message);

            if let Some(tc) = tool_calls {
                for tool_message in self.call_tools(&tc).await {
                    self.history.push(tool_message);
                }
            } else {
                return Ok(response.message);
            }
        }
    }


    async fn call_tools(&self, tool_calls: &Vec<ToolCall>) -> Vec<Message> {
        if let Some(avalible_tools) = &self.tools {
            let mut messages = vec![];
            for tool_call in tool_calls {
                for avalible_tool in avalible_tools {
                    if !avalible_tool.function.name.eq(&tool_call.function.name) {
                        continue;
                    }
                    match avalible_tool.execute(tool_call.function.arguments.clone()).await {
                        Ok(tool_result_content) => {
                            let response_tool_call_id = tool_call.id
                                .clone()
                                .unwrap_or_else(|| tool_call.function.name.clone());
    
    
                            messages.push(Message::tool(
                                tool_result_content,
                                response_tool_call_id, 
                            ));
                        }
                        Err(e) => {
                            eprintln!("Tool {} execution failed: {}", tool_call.function.name, e);
                            let error_content = format!("Error executing tool {}: {}", tool_call.function.name, e);
                            let response_tool_call_id = tool_call.id.clone().unwrap_or_else(|| tool_call.function.name.clone());
                            messages.push(Message::tool(
                                response_tool_call_id,
                                error_content,
                            ));
                        }
                    }
                }
            }
            messages
        } else {
            vec![Message::tool(
                "Tool",
                "Could not find tool with same name. Try again.",
            )]
        }
    }
}