use core::fmt;
use std::sync::Arc;
use std::{collections::HashMap, fs, path::Path};
use serde::de::DeserializeOwned;
use serde::Deserialize;
use tokio_stream::wrappers::ReceiverStream;
use futures::{stream::SelectAll, StreamExt};
use serde_json::{Error, Value};
use tokio::sync::mpsc::{self, Receiver, Sender};
use tokio::sync::Mutex;
use tracing::instrument;
use crate::models::agents::flow::flows::default_flow::default_flow;
use crate::models::configs::{ModelConfig, OllamaConfig, PromptConfig};
use crate::models::notification::NotificationContent;
use crate::util::templating::Template;

use crate::models::{AgentBuildError, AgentError};
use crate::{
    models::{agents::flow::invocation_flows::InternalFlow, notification::Notification}, 
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
pub struct Agent {
    pub name: String,
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
    pub stream: bool,
    pub notification_channel: Option<Sender<Notification>>,
    pub template: Option<Arc<Mutex<Template>>>,
    flow: InternalFlow,
    pub max_iterations: Option<usize>,
    pub clear_history_on_invoke: bool,

}

impl Agent {
    pub(crate) async fn try_new(
        name: String,
        model: &str,
        ollama_host: &str,
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
        stream: bool,
        top_k: Option<u32>,
        min_p: Option<f32>,
        notification_channel: Option<Sender<Notification>>,
        mcp_servers: Option<Vec<McpServerType>>,
        flow: InternalFlow,
        template: Option<Arc<Mutex<Template>>>,
        max_iterations: Option<usize>,
        clear_history_on_invoke: bool,

    ) -> Result<Self, AgentBuildError> {
        let history = vec![Message::system(system_prompt.to_string())];

        let mut agent = Self {
            name,
            model: model.into(),
            history,
            ollama_client: OllamaClient::new(ollama_host.to_string()),
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
            flow,
            tools: None,
            template,
            max_iterations,
            clear_history_on_invoke,
            stream,
        };

        agent.tools = agent.get_compiled_tools().await?;

        Ok(agent)
    }

    pub fn clear_history(&mut self) {
        self.history = vec![Message::system(self.system_prompt.clone())];
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
        // passed on creation of closures
        self.tools = self.get_compiled_tools().await?;
        Ok(r)
    }

    pub async fn notify(&self, content: NotificationContent) -> bool {
        if self.notification_channel.is_none() {
            return false;
        }
        let notification_channel = self.notification_channel.as_ref().unwrap();
        match notification_channel.send(Notification { agent: self.name.clone(), content, mcp_envelope: None }).await {
            Ok(_) => true,
            Err(e) => {
                tracing::error!(error = %e, "Failed sending notification");
                false
            },
        }
    }

    pub fn forward_notifications(
        &self,
        mut from_channel: Receiver<Notification>
    ) {
        if let Some(notification_channel) = &self.notification_channel {
            let to_sender = notification_channel.clone();
            tokio::spawn(async move {
                while let Some(msg) = from_channel.recv().await {

                    if to_sender.send(msg.unwrap()).await.is_err() {
                        break;
                    }
                }
            });    
        }
    }

    /// Merge any number of `Receiver<Notification>` streams into one,
    /// and forward *all* messages (in the order they arrive) into
    /// this agent’s `notification_channel`.
    pub fn forward_multiple_notifications<I>(&self, channels: I)
    where
        I: IntoIterator<Item = Receiver<Notification>>,
    {
        let to_sender = match &self.notification_channel {
            Some(s) => s.clone(),
            None    => return,
        };

        let mut merged = SelectAll::new();
        for rx in channels {
            let stream = ReceiverStream::new(rx)
                .map(|notif| notif);
            merged.push(stream);
        }

        tokio::spawn(async move {
            while let Some(notification) = merged.next().await {
                if to_sender.send(notification).await.is_err() {
                    break;
                }
            }
        });
    }

    pub async fn get_compiled_tools(&self) -> Result<Option<Vec<Tool>>, AgentBuildError> {
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

    pub async fn get_compiled_mcp_tools(&self) -> Result<Option<Vec<Tool>>, AgentBuildError> {
        let mut running_tools: Option<Vec<Tool>> = None;
        if let Some(mcp_servers) = &self.mcp_servers {
            for mcp_server in mcp_servers {
                let mcp_tools = match get_mcp_tools(mcp_server.clone(), self.notification_channel.clone()).await {
                    Ok(t) => t,
                    Err(e) => return Err(AgentBuildError::McpError(e)),
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




    #[instrument(level = "debug", skip(self, prompt), fields(agent_name = %self.name))]
    pub async fn invoke_flow<T>(&mut self, prompt: T) -> Result<Message, AgentError>
    where
        T: Into<String>,
    {
        self.execute_invocation(prompt.into()).await
    }

    #[instrument(level = "debug", skip(self, prompt), fields(agent_name = %self.name))]
    pub async fn invoke_flow_structured_output<T, O>(&mut self, prompt: T) -> Result<O, AgentError>
    where
        T: Into<String>,
        O: DeserializeOwned
    {
        let response = self.execute_invocation(prompt.into()).await?;
        let Some(json) = response.content else {
            return Err(AgentError::RuntimeError("Agent did not produce answer".into()))
        };
        let out: O = serde_json::from_str(&json)
            .map_err(|e| AgentError::Deserialization(e))?; 
        Ok(out)
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
        

        self.execute_invocation(prompt).await
    }

    #[instrument(level = "debug", skip(self, prompt))]
    async fn execute_invocation(&mut self, prompt: String) -> Result<Message, AgentError>  {
        let flow_to_run = self.flow.clone();

        if self.clear_history_on_invoke {
            self.clear_history();
        }

        match flow_to_run {
            InternalFlow::Default => default_flow(self, prompt).await,
            InternalFlow::Custom(custom_flow_fn) => (custom_flow_fn)(self, prompt).await,
        }
    }


    pub fn export_ollama_config(&self) -> OllamaConfig {
        OllamaConfig { ollama_url: Some(self.ollama_client.base_url.clone()) }
    }

    pub fn export_model_config(&self) -> ModelConfig {
        ModelConfig { 
            model: Some(self.model.clone()), 
            temperature: self.temperature, 
            top_p: self.top_p, 
            presence_penalty: self.presence_penalty, 
            frequency_penalty: self.frequency_penalty,
            num_ctx: self.num_ctx, 
            repeat_last_n: self.repeat_last_n, 
            repeat_penalty: self.repeat_penalty, 
            seed: self.seed, 
            stop: self.stop.clone(), 
            num_predict: self.num_predict, 
            top_k: self.top_k, 
            min_p: self.min_p 
        }
    }

    pub async fn export_prompt_config(&self) -> Result<PromptConfig, Error> {
        let template = if let Some(t) = self.template.clone() {
            Some(t.lock().await.clone())
        } else {
            None
        };

        

        let response_format = if let Some(p) = self.response_format.clone() {
            Some(serde_json::to_string(&p)?)
        } else {
            None
        };
        Ok(PromptConfig {
            template,
            system_prompt: Some(self.system_prompt.clone()),
            tools: self.tools.clone(),
            response_format,
            mcp_servers: self.mcp_servers.clone(),
            stop_prompt: self.stop_prompt.clone(),
            stopword: self.stopword.clone(),
            strip_thinking: Some(self.strip_thinking),
            max_iterations: self.max_iterations,
            clear_histroy_on_invoke: Some(self.clear_history_on_invoke),
            stream: self.stream
        })
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
