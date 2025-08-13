use std::sync::Arc;

use serde_json::Value;
use tokio::{process::Command, sync::{mpsc::Sender, Mutex}};
use tracing::{info, instrument, trace};

use crate::{Tool, notifications::{Notification, NotificationContent}, ToolBuilder, ToolExecutionError};

use super::error::McpIntegrationError;
use rmcp::{model::{CallToolRequestParam, CallToolResult, JsonObject}, service::RunningService, transport::{ConfigureCommandExt, SseClientTransport, StreamableHttpClientTransport, TokioChildProcess}, ClientHandler, ServiceExt};
use crate::AsyncToolFn;


#[derive(Debug, Clone, PartialEq, Eq)]
pub enum McpServerType {
    Sse(String),
    Stdio(String),
    StreamableHttp(String)
}

impl McpServerType {
    pub fn sse<S: Into<String>>(url: S) -> Self { McpServerType::Sse(url.into()) }
    pub fn stdio<S: Into<String>>(cmd: S) -> Self { McpServerType::Stdio(cmd.into()) }
    pub fn streamable_http<S: Into<String>>(url: S) -> Self { McpServerType::StreamableHttp(url.into()) }
}


pub type ArcMcpClient = RunningService<rmcp::RoleClient, AgentMcpHandler>;
pub type McpClient = Arc<Mutex<ArcMcpClient>>;


#[derive(Clone)]
pub struct AgentMcpHandler {
    /// The channel to send notifications back to the main agent.
    agent_notification_tx: Option<Sender<Notification>>,
}

impl ClientHandler for AgentMcpHandler {
    async fn on_progress(
        &self,
        params: rmcp::model::ProgressNotificationParam,
        _context: rmcp::service::NotificationContext<rmcp::RoleClient>,
    ) {
        trace!("Received progress notification: {:?}", params);
        if self.agent_notification_tx.is_none() {
            return;
        }
        let tx = self.agent_notification_tx.clone().unwrap();
        let notification_string = serde_json::to_string(&params)
            .unwrap_or_else(|e| format!("Failed to serialize MCP notification: {e}"));

        let agent_notification = Notification { 
            agent: "MCP".to_string(), 
            content: NotificationContent::McpToolNotification(notification_string),
            mcp_envelope: None,
        }.unwrap();

        if tx.send(agent_notification).await.is_err() {
            tracing::warn!("Agent notification channel closed. Cannot forward MCP notification.");
        }
    }

}


#[instrument(level = "debug", skip(mcp_server_type, notification_channel))]
pub async fn get_mcp_tools(mcp_server_type: McpServerType, notification_channel: Option<Sender<Notification>>) -> Result<Vec<Tool>, McpIntegrationError> {
    
    let (mcp_client, mcp_raw_tools) = match mcp_server_type {
        McpServerType::Sse(url) => get_mcp_sse_tools(url, notification_channel).await?,
        McpServerType::StreamableHttp(url) => get_mcp_streamable_http_tools(url, notification_channel).await?,
        McpServerType::Stdio(command) => get_mcp_stdio_tools(command, notification_channel).await?,
    };

    

    trace!(
        "Discovered {} raw tools from MCP server. Converting...",
        mcp_raw_tools.len()
    );
    let mut agent_tools = Vec::new();

    for mcp_tool_def in mcp_raw_tools {
        

        let arc_mcp_client = Arc::clone(&mcp_client);
        let tool_namme_cow = mcp_tool_def.name.clone();


        // MCP tool executor closure
        let executor: AsyncToolFn = Arc::new(move |args: Value| {
            
            let mcp_client_ref = Arc::clone(&arc_mcp_client);
            let tool_name = tool_namme_cow.clone().into_owned();
            
            Box::pin(async move {
                let inner_mcp_client = mcp_client_ref.lock().await;

                // call remote tool
                let result = match inner_mcp_client
                    .call_tool(CallToolRequestParam {
                        name: tool_name.clone().into(),
                        arguments: serde_json::json!(args).as_object().cloned(),
                    })
                    .await
                {
                    Ok(result) => result,
                    Err(e) => return Err(ToolExecutionError::ExecutionFailed(format!(
                        "MCP tool '{tool_name}' execution failed: {e}"
                    ))),
                };

                let CallToolResult {content, is_error } = result;

                if let Some(true) = is_error {
                    return Err(ToolExecutionError::ExecutionFailed(format!("tool call failed, mcp call error: {content:#?}")));
                }

                
                let mut out_result = "".to_string();
                for content in content.iter() {
                    if let Some(content_text) = content.as_text() {
                        out_result = format!("{}\n{}", out_result, content_text.text);
                    }
                }
                Ok(out_result.to_string())
            })
        });

        let tool_name = mcp_tool_def.name.clone().into_owned();
        let tool_desciption = match mcp_tool_def.description {
            Some(d) => d.into_owned(),
            None => return Err(McpIntegrationError::ToolConversion("No description".to_string())),
        };

        let mut tool_builer = ToolBuilder::new()
            .function_name(tool_name)
            .function_description(tool_desciption)
            .executor(executor);

        // create tool description
        
        let input_schema_json_obj: &JsonObject = &mcp_tool_def.input_schema;

        if let Some(Value::Object(properties_map)) = input_schema_json_obj.get("properties") {
            for (prop_name, prop_schema_value) in properties_map {
                if let Value::Object(prop_details) = prop_schema_value {
                    let property_type = prop_details
                        .get("type")
                        .and_then(Value::as_str)
                        .unwrap_or("string")
                        .to_string();
                    
                    let description = prop_details
                        .get("description")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();
                    
                    tool_builer = tool_builer.add_required_property(
                        prop_name.clone(),
                        property_type,
                        description,
                    );
                }
            }
        }

        let created_tool = match tool_builer.build() {
            Ok(tool) => tool,
            Err(e) => return Err(McpIntegrationError::ToolConversion(e.to_string())),
        };

       
        agent_tools.push(created_tool);
        info!("[MCP] Registered agent tool for MCP action: {}", mcp_tool_def.name);
    }

    Ok(agent_tools)
}



pub async fn get_mcp_sse_tools<T>(url: T, notification_channel: Option<Sender<Notification>>) -> Result<(McpClient, Vec<rmcp::model::Tool>), McpIntegrationError> where T: Into<String> {
    let transport = match  SseClientTransport::start(url.into()).await {
        Ok(t) => t,
        Err(e) => return Err(McpIntegrationError::Connection(e.to_string())),
    };
    
    let handler = AgentMcpHandler {
        agent_notification_tx: notification_channel,
    };

    let client = match handler.serve(transport).await {
        Ok(c) =>c,
        Err(e) => return Err(McpIntegrationError::Connection(e.to_string())),
    };

    let tool_list = match client.list_tools(Default::default()).await {
        Ok(l) => l,
        Err(e) => return Err(McpIntegrationError::Discovery(e.to_string())),
    };
    Ok((Arc::new(Mutex::new(client)), tool_list.tools))
}


pub async fn get_mcp_streamable_http_tools<T>(url: T, notification_channel: Option<Sender<Notification>>) -> Result<(McpClient, Vec<rmcp::model::Tool>), McpIntegrationError> where T: Into<String> {
    let transport = StreamableHttpClientTransport::from_uri(url.into());

    let handler = AgentMcpHandler {
        agent_notification_tx: notification_channel,
    };

    let client = handler.serve(transport).await.map_err(|e| McpIntegrationError::Connection(e.to_string()))?;
    let tool_list = client.list_tools(Default::default()).await.map_err(|e| McpIntegrationError::Discovery(e.to_string()))?;
    
    Ok((Arc::new(Mutex::new(client)), tool_list.tools))
}


pub async fn get_mcp_stdio_tools<T>(full_command: T, notification_channel: Option<Sender<Notification>>) -> Result<(McpClient, Vec<rmcp::model::Tool>), McpIntegrationError> where T: Into<String> {
    let full_command_string = full_command.into();
    let mut command_args = full_command_string.split(" ");
    let first = command_args.next();
    if first.is_none() {
        return Err(McpIntegrationError::Connection("Invalid command.".into()));
    }
    let transport = match TokioChildProcess::new(Command::new(first.unwrap()).configure(
        |cmd| {
            for arg in command_args {
                cmd.arg(arg);
            }
        },
    )) {
        Ok(t) =>t,
        Err(e) => return Err(McpIntegrationError::Connection(e.to_string())),
    };

    let handler = AgentMcpHandler {
        agent_notification_tx: notification_channel,
    };

    let client = match handler.serve(transport)
        .await {
            Ok(c) =>c,
            Err(e) => return Err(McpIntegrationError::Connection(e.to_string())),
        };

    let tool_list = match client.list_tools(Default::default()).await {
        Ok(l) => l,
        Err(e) => return Err(McpIntegrationError::Discovery(e.to_string())),
    };
    Ok((Arc::new(Mutex::new(client)), tool_list.tools))
}