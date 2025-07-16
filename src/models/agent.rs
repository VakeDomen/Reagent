use core::fmt;
use std::{fs, path::Path};

use serde_json::Value;
use tokio::sync::mpsc::{self, Sender};
use tracing::instrument;

use crate::{
    models::{agent, invocation::{invocation_handler::{FlowFn, FlowFuture, InvokeFn}, invocation_util::{call_model, call_tools, generate_llm_request}, simple_loop::simple_loop_invoke}, notification::Notification}, 
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


#[derive( Clone)]
pub struct Agent {
    pub model: String,
    pub history: Vec<Message>,

    pub local_tools: Option<Vec<Tool>>,
    pub mcp_servers: Option<Vec<McpServerType>>,

    pub tools: Option<Vec<Tool>>,

    pub response_format: Option<Value>,
    pub(crate) ollama_client: OllamaClient,
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
    invoke_fn: FlowFn,
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
        mcp_servers: Option<Vec<McpServerType>>,
        invoke_fn: FlowFn,

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
            invoke_fn,
            tools: None,
        }
    }

    pub fn clear_history(&mut self) {
        self.history = vec![Message::system(self.system_prompt.clone())];
    }

    #[instrument(level = "debug", skip(self, prompt))]
    pub async fn invoke_flow<T>(&mut self, prompt: T) -> Result<Message, AgentError>
    where
        T: Into<String>,
    {
        (self.invoke_fn)(self, prompt.into()).await
        // self.history.push(Message::user(prompt.into()));
        // let running_tools = self.get_compiled_tools().await?;

        // loop {

        //     let request = generate_llm_request(self, running_tools.clone());
        //     let mut response = call_model(self, request).await?;

        //     if self.strip_thinking {
        //         if response.message.content.clone().unwrap().contains("</think>") {
        //             response.message.content = Some(response.message
        //                 .content
        //                 .unwrap()
        //                 .split("</think>")
        //                 .nth(1)
        //                 .unwrap()
        //                 .to_string()
        //             );
        //         }
        //     }


        //     self.history.push(response.message.clone());

        //     if let Some(tc) = response.message.tool_calls {
        //         for tool_message in call_tools(self, &tc, &running_tools).await {
        //             self.history.push(tool_message);
        //         }
        //     } else {
        //         if let Some(stopword) = &self.stopword {
        //             if response.message.clone().content.unwrap().contains(stopword) {
        //                 self.notify(Notification::Done(true)).await;
        //                 return Ok(response.message);
        //             } else if let Some(stop_prompt) = &self.stop_prompt {
        //                 self.history.push(Message::tool( stop_prompt, "0"));
        //             }
        //         } else {
        //             self.notify(Notification::Done(true)).await;
        //             return Ok(response.message);
        //         }
        //     } 
        // }
    }

    

    pub fn save_history<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let json_string = serde_json::to_string_pretty(&self.history)?;
        fs::write(path, json_string)?;
        Ok(())
    }

    pub async fn new_notification_channel(&mut self) -> Result<mpsc::Receiver<Notification>, AgentError> {
        let (s, r) = mpsc::channel::<Notification>(100);
        self.notification_channel = Some(s);
        // have to reset mcp tools for notifications as the channel is 
        // passed on creation of closure
        self.tools = None;
        Ok(r)
    }

    pub(crate) async fn notify(&self, msg: Notification) -> bool {
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

        match self.get_compiled_mcp_tools().await {
            Ok(tools_option) => if let Some(mcp_tools) = tools_option {
                match running_tools.as_mut() {
                    Some(t) => for mcpt in mcp_tools { t.push(mcpt); },
                    None => if mcp_tools.len() > 0 {
                        running_tools = Some(mcp_tools)
                    },
                }
            },
            Err(e) => return Err(e),
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



impl fmt::Debug for Agent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Agent")
            .field("model", &self.model)
            .field("history", &self.history)
            .field("local_tools", &self.local_tools)
            .field("response_format", &self.response_format)
            .field("ollama_client", &self.ollama_client)
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