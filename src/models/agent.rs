use std::{fs, path::Path};

use serde_json::Value;
use tokio::sync::mpsc::{self, Sender};
use tracing::instrument;

use crate::{
    models::notification::Notification, 
    services::{
        mcp::mcp_tool_builder::get_mcp_tools, 
        ollama::{
            client::OllamaClient, 
            models::{
                base::{BaseRequest, Message, OllamaOptions}, 
                chat::{ChatRequest, ChatResponse}, 
                tool::{Tool, ToolCall}}
            }
        }, 
        McpServerType
    };

use super::AgentError;


#[derive(Debug, Clone)]
pub struct Agent {
    pub model: String,
    pub history: Vec<Message>,

    pub local_tools: Option<Vec<Tool>>,

    pub response_format: Option<Value>,
    ollama_client: OllamaClient,
    pub system_prompt: String,
    pub stop_prompt: Option<String>,
    pub stopword: Option<String>,
    pub strip_thinking: bool,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub num_ctx: Option<u32>,
    pub repeat_last_n: Option<i32>,
    pub repeat_penalty: Option<f32>,
    pub seed: Option<i32>,
    pub stop: Option<String>,
    pub num_predict: Option<i32>,
    pub top_k: Option<u32>,
    pub min_p: Option<f32>,
    pub notification_channel: Option<Sender<Notification>>,
    pub mcp_servers: Option<Vec<McpServerType>>,
}

impl Agent {
    pub(crate) fn new(
        model: &str,
        ollama_host: &str,
        ollama_port: u16,
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
        top_k: Option<u32>,
        min_p: Option<f32>,
        notification_channel: Option<Sender<Notification>>,
        mcp_servers: Option<Vec<McpServerType>>
    ) -> Self {
        let base_url = format!("{}:{}", ollama_host, ollama_port);
        let history = vec![Message::system(system_prompt.to_string())];

        Self {
            model: model.into(),
            history,
            ollama_client: OllamaClient::new(base_url),
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
            notification_channel,
            mcp_servers,
            local_tools,
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
        let running_tools = self.get_compiled_tools().await?;

        loop {

            let request = self.generate_llm_request(running_tools.clone());
            let mut response = self.call_model(request).await?;

            if self.strip_thinking {
                if response.message.content.clone().unwrap().contains("</think>") {
                    response.message.content = Some(response.message
                        .content
                        .unwrap()
                        .split("</think>")
                        .nth(1)
                        .unwrap()
                        .to_string()
                    );
                }
            }


            self.history.push(response.message.clone());

            if let Some(tc) = response.message.tool_calls {
                for tool_message in self.call_tools(&tc, &running_tools).await {
                    self.history.push(tool_message);
                }
            } else {
                if let Some(stopword) = &self.stopword {
                    if response.message.clone().content.unwrap().contains(stopword) {
                        self.notify(Notification::Done(true)).await;
                        return Ok(response.message);
                    } else if let Some(stop_prompt) = &self.stop_prompt {
                        self.history.push(Message::tool( stop_prompt, "0"));
                    }
                } else {
                    self.notify(Notification::Done(true)).await;
                    return Ok(response.message);
                }
            } 
        }
    }

    fn generate_llm_request(&self, tools: Option<Vec<Tool>>) -> ChatRequest {
        ChatRequest {
            base: BaseRequest {
                model: self.model.clone(),
                format: self.response_format.clone(),
                options:  Some(OllamaOptions {
                    num_ctx: self.num_ctx,
                    repeat_last_n: self.repeat_last_n,
                    repeat_penalty: self.repeat_penalty,
                    temperature: self.temperature,
                    seed: self.seed,
                    stop: self.stop.clone(),
                    num_predict: self.num_predict,
                    top_k: self.top_k,
                    top_p: self.top_p,
                    min_p: self.min_p,
                    presence_penalty: self.presence_penalty,
                    frequency_penalty: self.frequency_penalty,
                }),
                stream: Some(false), 
                keep_alive: Some("5m".to_string()),
            },
            messages: self.history.clone(),
            tools: tools, 
        }
    }

    async fn call_model(&self, request: ChatRequest) -> Result<ChatResponse, AgentError> {
        self.notify(Notification::PromptRequest(request.clone())).await;
        match self.ollama_client.chat(request).await {
            Ok(resp) => {
                self.notify(Notification::PromptSuccessResult(resp.clone())).await;
                Ok(resp)
            }
            Err(e) => {
                self.notify(Notification::PromptErrorResult(e.to_string())).await;
                Err(e.into())
            } 
        }
    }

    async fn call_tools(&self, tool_calls: &Vec<ToolCall>, running_tools: &Option<Vec<Tool>>) -> Vec<Message> {
        if let Some(avalible_tools) = running_tools {
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
                    self.notify(Notification::ToolCallRequest(tool_call.clone())).await;

                    match avalible_tool.execute(tool_call.function.arguments.clone()).await {
                        Ok(tool_result_content) => {
                            let response_tool_call_id = tool_call.id
                                .clone()
                                .unwrap_or_else(|| tool_call.function.name.clone());
    
                            self.notify(Notification::ToolCallSuccessResult(tool_result_content.clone())).await;
                            messages.push(Message::tool(
                                tool_result_content,
                                response_tool_call_id, 
                            ));
                        }
                        Err(e) => {
                            tracing::error!(error = %e, "Tool execution failed");
                            let error_content = format!("Error executing tool {}: {}", tool_call.function.name, e);
                            let response_tool_call_id = tool_call.id.clone().unwrap_or_else(|| tool_call.function.name.clone());
                            
                            self.notify(Notification::ToolCallErrorResult(e.to_string())).await;
                            messages.push(Message::tool(
                                error_content,
                                response_tool_call_id,
                            ));
                        }
                    }
                }
                if !tool_found {
                    tracing::error!("No corresponding tool found.");
                    let message = format!("Could not find tool: {}", tool_call.function.name);
                    self.notify(Notification::ToolCallErrorResult(message.clone())).await;
                    messages.push(Message::tool(
                        message, 
                        "0"
                    ));
                }

            }
            messages
        } else {
            tracing::error!("No tools specified");
            self.notify(Notification::ToolCallErrorResult("Empty tool call".to_string())).await;
            vec![Message::tool(
                "If you want to use a tool specifiy the name of the avalible tool.",
                "Tool",
            )]
        }
    }

    pub fn save_history<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let json_string = serde_json::to_string_pretty(&self.history)?;
        fs::write(path, json_string)?;
        Ok(())
    }

    pub async fn new_notification_channel(&mut self) -> Result<mpsc::Receiver<Notification>, AgentError> {
        let (s, r) = mpsc::channel::<Notification>(100);
        self.notification_channel = Some(s);
        Ok(r)
    }

    async fn notify(&self, msg: Notification) -> bool {
        if let None = self.notification_channel {
            return false;
        }
        let notification_channel = self.notification_channel.as_ref().unwrap();
        match notification_channel.send(msg).await {
            Ok(_) => true,
            Err(e) => {
                tracing::error!(error = %e, "Failed sending notification");
                false
            },
        }
    }

    pub async fn get_compiled_tools(&self) -> Result<Option<Vec<Tool>>, AgentError> {
        let mut running_tools = self.local_tools.clone();
        if let Ok(Some(mcp_tools)) = self.get_compiled_mcp_tools().await {
            match running_tools.as_mut() {
                Some(t) => for mcpt in mcp_tools { t.push(mcpt); },
                None => if mcp_tools.len() > 0 {
                    running_tools = Some(mcp_tools)
                },
            }
        }
        Ok(running_tools)
    }

    pub async fn get_compiled_mcp_tools(&self) -> Result<Option<Vec<Tool>>, AgentError> {
        let mut running_tools: Option<Vec<Tool>> = None;
        if let Some(mcp_servers) = &self.mcp_servers {
            for mcp_server in mcp_servers {
                let mcp_tools = match get_mcp_tools(mcp_server.clone(), self.notification_channel.clone()).await {
                    Ok(t) => t,
                    Err(e) => return Err(e.into()),
                };
    
                match running_tools.as_mut() {
                    Some(t) => for mcpt in mcp_tools { t.push(mcpt); },
                    None => if mcp_tools.len() > 0 {
                        running_tools = Some(mcp_tools)
                    },
                }
            }
        }
        Ok(running_tools)
    } 

}