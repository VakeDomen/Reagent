use core::fmt;
use std::sync::Arc;
use std::{collections::HashMap, fs, path::Path};

use serde_json::Value;
use tokio::sync::mpsc::{self, Sender};
use tokio::sync::Mutex;
use tracing::instrument;
use crate::models::agents::flow::flows::plan_and_execute::{self, plan_and_execute_invoke};
use crate::models::agents::flow::util::templating::Template;

use crate::models::AgentError;
use crate::{
    models::{agents::flow::{invocation_flows::InternalFlow, flows::simple_loop::simple_loop_invoke}, notification::Notification}, 
    services::{
        mcp::mcp_tool_builder::get_mcp_tools, 
        ollama::{
            client::OllamaClient, 
            models::{
                base::Message, 
                tool::Tool}
            }
        }, 
        McpServerType
    };



#[derive(Clone)]
pub struct Node{
    pub model: String,

    pub local_tools: Option<Vec<Tool>>,
    pub mcp_servers: Option<Vec<McpServerType>>,

    pub tools: Option<Vec<Tool>>,

    pub response_format: Option<Value>,
    pub(crate) ollama_client: OllamaClient,
    pub system_prompt: String,
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
    pub template: Option<Arc<Mutex<Template>>>,
}

impl Node {
    pub(crate) fn new(
        model: &str,
        ollama_host: &str,
        ollama_port: u16,
        system_prompt: &str,
        local_tools: Option<Vec<Tool>>,
        response_format: Option<Value>,
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
        template: Option<Arc<Mutex<Template>>>,

    ) -> Self {
        let base_url = format!("{ollama_host}:{ollama_port}");

        Self {
            model: model.into(),
            ollama_client: OllamaClient::new(base_url),
            response_format,
            system_prompt: system_prompt.into(),
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
            tools: None,
            template,
        }
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
        if self.notification_channel.is_none() {
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
                    None => if !mcp_tools.is_empty() {
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
                    None => if !mcp_tools.is_empty() {
                        running_tools = Some(mcp_tools)
                    },
                }
            }
        }
        Ok(running_tools)
    } 




    #[instrument(level = "debug", skip(self, prompt))]
    pub async fn invoke_flow<T>(&mut self, prompt: T) -> Result<Message, AgentError>
    where
        T: Into<String>,
    {
        self.execute_invocation(prompt.into()).await
    }

    #[instrument(level = "debug", skip(self, template_data))]
    pub async fn invoke_flow_with_template<K, V>(&mut self, template_data: HashMap<K, V>) -> Result<Message, AgentError>
    where
        K: Into<String>,
        V: Into<String>,
    {
        let Some(template) = &self.template else {
            return Err(AgentError::RuntimeError("No template defined".into()));
        };

        let string_map: HashMap<String, String> = template_data
            .into_iter()
            .map(|(k, v)| (k.into(), v.into()))
            .collect();

        let prompt = {
            template
                .lock()
                .await
                .compile(&string_map)
                .await
        };
        

        self.execute_invocation(prompt.into()).await
    }


    async fn execute_invocation(&mut self, prompt: String) -> Result<Message, AgentError>  {

        todo!()
    }
}

impl fmt::Debug for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Node")
            .field("model", &self.model)
            .field("local_tools", &self.local_tools)
            .field("response_format", &self.response_format)
            .field("ollama_client", &self.ollama_client)
            .field("system_prompt", &self.system_prompt)
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