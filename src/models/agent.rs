use serde_json::Value;
use tracing::instrument;

use crate::services::ollama::{client::OllamaClient, models::{base::{BaseRequest, Message}, chat::{ChatRequest, ChatResponse}, tool::{Tool, ToolCall}}};

use super::AgentError;


#[derive(Debug, Clone)]
pub struct Agent {
    pub model: String,
    pub history: Vec<Message>,
    pub tools: Option<Vec<Tool>>,
    pub response_format: Option<Value>,
    ollama_client: OllamaClient,
    pub system_prompt: String,
    pub stop_prompt: Option<String>,
    pub stopword: Option<String>,
    pub strip_thinking: bool,
}

impl Agent {
    pub(crate) fn new(
        model: &str,
        ollama_host: &str,
        ollama_port: u16,
        system_prompt: &str,
        tools: Option<Vec<Tool>>,
        response_format: Option<Value>,
        stop_prompt: Option<String>,
        stopword: Option<String>,
        strip_thinking: bool,
    ) -> Self {
        let base_url = format!("{}:{}", ollama_host, ollama_port);
        let history = vec![Message::system(system_prompt.to_string())];

        Self {
            model: model.into(),
            history,
            ollama_client: OllamaClient::new(base_url),
            tools,
            response_format,
            system_prompt: system_prompt.into(),
            stop_prompt,
            stopword,
            strip_thinking
        }
    }

    pub fn clear_history(&mut self) {
        self.history = vec![Message::system(self.system_prompt.clone())];
    }

    #[instrument(level = "debug", skip(self, prompt))]
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
            let mut message = response.message.clone();
           
            let tool_calls = message.tool_calls.clone();

            if self.strip_thinking {
                if message.content.clone().unwrap().contains("</think>") {
                    message.content = Some(message
                        .content
                        .unwrap()
                        .split("</think>")
                        .nth(1)
                        .unwrap()
                        .to_string()
                    );
                }
            }

            self.history.push(message);

            if let Some(tc) = tool_calls {
                for tool_message in self.call_tools(&tc).await {
                    self.history.push(tool_message);
                }
            } else {
                if let Some(stopword) = &self.stopword {
                    if response.message.clone().content.unwrap().contains(stopword) {
                        return Ok(response.message);
                    } else if let Some(stop_prompt) = &self.stop_prompt {
                        self.history.push(Message::tool( stop_prompt, "0"));
                    }
                } else {
                    return Ok(response.message);
                }
            } 
        }
    }


    async fn call_tools(&self, tool_calls: &Vec<ToolCall>) -> Vec<Message> {
        if let Some(avalible_tools) = &self.tools {
            let mut messages = vec![];
            for tool_call in tool_calls {
                tracing::info!(
                    target: "tool",                    
                    tool = %tool_call.function.name,
                    id   = ?tool_call.id,
                    args = ?tool_call.function.arguments,
                    "executing tool call"
                );
                let mut tool_found = false;
                for avalible_tool in avalible_tools {
                    if !avalible_tool.function.name.eq(&tool_call.function.name) {
                        continue;
                    }

                    tool_found = true;
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
                            tracing::error!(error = %e, "Tool execution failed");
                            let error_content = format!("Error executing tool {}: {}", tool_call.function.name, e);
                            let response_tool_call_id = tool_call.id.clone().unwrap_or_else(|| tool_call.function.name.clone());
                            messages.push(Message::tool(
                                error_content,
                                response_tool_call_id,
                            ));
                        }
                    }
                }
                if !tool_found {
                    tracing::error!("No corresponding tool found.");
                    messages.push(Message::tool(
                        format!("Could not find tool: {}", tool_call.function.name), 
                        "0"
                    ));
                }

            }
            messages
        } else {
            tracing::error!("No tools specified");
            vec![Message::tool(
                "If you want to use a tool specifiy the name of the avalible tool.",
                "Tool",
            )]
        }
    }
}