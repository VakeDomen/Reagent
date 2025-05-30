use std::sync::Arc;

use serde_json::Value;
use tokio::{process::Command, sync::Mutex};

use crate::{services::ollama::models::tool::Tool, ToolBuilder, ToolExecutionError};

use super::error::McpIntegrationError;
use rmcp::{model::{CallToolRequestParam, ClientCapabilities, ClientInfo, Implementation, JsonObject}, service::RunningService, transport::{ConfigureCommandExt, SseClientTransport, StreamableHttpClientTransport, TokioChildProcess}, ServiceError, ServiceExt};
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



pub type McpClient = Arc<Mutex<McpClientType>>;



pub enum McpClientType {
    SseClient(ArcMcpSseClient),
    StdioClient(ArcMcpStdioClient),
    StreamableHttp(ArcMcpStreamableHttpClient)
}

impl McpClientType {
    pub fn call_tool(&mut self, param: CallToolRequestParam) -> impl Future<Output = Result<rmcp::model::CallToolResult, ServiceError>> {
        match self {
            McpClientType::SseClient(running_service) => running_service.call_tool(param),
            McpClientType::StdioClient(running_service) => running_service.call_tool(param),
            McpClientType::StreamableHttp(running_service) => running_service.call_tool(param),
        }
    }
}

pub async fn get_mcp_tools(mcp_server_type: McpServerType) -> Result<Vec<Tool>, McpIntegrationError> {
    
    let (mcp_client_arc, mcp_raw_tools) = match mcp_server_type {
        McpServerType::Sse(url) => get_mcp_sse_tools(url).await?,
        McpServerType::StreamableHttp(url) => get_mcp_streamable_http_tools(url).await?,
        McpServerType::Stdio(command) => get_mcp_stdio_tools(command).await?,
    };

    

    println!(
        "[MCP] Discovered {} raw tools from MCP server. Converting...",
        mcp_raw_tools.len()
    );
    let mut agent_tools = Vec::new();

    for mcp_tool_def in mcp_raw_tools {
        

        let client_for_executor = Arc::clone(&mcp_client_arc);
        let action_name_for_executor = mcp_tool_def.name.clone();


        let executor: AsyncToolFn = Arc::new(move |args: Value| {
            let client_captured_arc = Arc::clone(&client_for_executor);
            let action_name_captured = action_name_for_executor.clone().into_owned();
            Box::pin(async move {
                let mut client_captured = client_captured_arc.lock().await;
                match client_captured
                    .call_tool(CallToolRequestParam {
                        name: action_name_captured.clone().into(),
                        arguments: serde_json::json!(args).as_object().cloned(),
                    })
                    .await
                {
                    Ok(result) => {
                        if result.is_error.is_some_and(|b| b) {
                            Err(ToolExecutionError::ExecutionFailed("tool call failed, mcp call error".into()))
                        } else {
                            let mut out_result = "".to_string();
                            for content in result.content.iter() {
                                if let Some(content_text) = content.as_text() {
                                    out_result = format!("{}\n{}", out_result, content_text.text);
                                }
                            }
                            Ok(out_result.to_string())
                        }
                    }
                    Err(e) => Err(ToolExecutionError::ExecutionFailed(format!(
                        "MCP tool '{}' execution failed: {}",
                        action_name_captured, e.to_string()
                    ))),
                }
            })
        });

        let tool_name = mcp_tool_def.name.clone().into_owned();
        let tool_desciption = match mcp_tool_def.description {
            Some(d) => d.into_owned(),
            None => return Err(McpIntegrationError::ToolConversionError("No description".to_string())),
        };

        let mut tool_builer = ToolBuilder::new()
            .function_name(tool_name)
            .function_description(tool_desciption)
            .executor(executor);
 
        let input_schema_json_obj: &JsonObject = &*mcp_tool_def.input_schema;

        if let Some(Value::Object(properties_map)) = input_schema_json_obj.get("properties") {
            for (prop_name, prop_schema_value) in properties_map {
                if let Value::Object(prop_details) = prop_schema_value {
                    let property_type = prop_details
                        .get("type")
                        .and_then(Value::as_str)
                        .unwrap_or("string") // Default to string if type not specified
                        .to_string();
                    
                    let description = prop_details
                        .get("description")
                        .and_then(Value::as_str)
                        .unwrap_or("") // Default to empty string for description
                        .to_string();
                    
                    tool_builer = tool_builer.add_property(
                        prop_name.clone(), // prop_name is &String from properties_map keys
                        property_type,
                        description,
                    );
                }
            }
        }

        if let Some(Value::Array(required_array)) = input_schema_json_obj.get("required") {
            for req_val in required_array {
                if let Value::String(req_name) = req_val {
                    tool_builer = tool_builer.add_required_property(req_name.clone());
                }
            }
        }

        let created_tool = match tool_builer.build() {
            Ok(tool) => tool,
            Err(e) => return Err(McpIntegrationError::ToolConversionError(e.to_string())),
        };

       
        agent_tools.push(created_tool);
        println!("[MCP] Registered agent tool for MCP action: {}", mcp_tool_def.name);
    }

    Ok(agent_tools)
}


pub type ArcMcpSseClient = RunningService<rmcp::RoleClient, rmcp::model::InitializeRequestParam>;
pub type ArcMcpStreamableHttpClient = RunningService<rmcp::RoleClient, rmcp::model::InitializeRequestParam>;
pub type ArcMcpStdioClient = RunningService<rmcp::RoleClient, ()>;

pub async fn get_mcp_sse_tools<T>(url: T) -> Result<(McpClient, Vec<rmcp::model::Tool>), McpIntegrationError> where T: Into<String> {
    let transport = match  SseClientTransport::start(url.into()).await {
        Ok(t) => t,
        Err(e) => return Err(McpIntegrationError::ConnectionError(e.to_string())),
    };
    let client_info = ClientInfo {
        protocol_version: Default::default(),
        capabilities: ClientCapabilities::default(),
        client_info: Implementation {
            name: "test sse client".to_string(),
            version: "0.0.1".to_string(),
        },
    };
    let client = match client_info.serve(transport).await {
        Ok(c) =>c,
        Err(e) => return Err(McpIntegrationError::ConnectionError(e.to_string())),
    };

    let tool_list = match client.list_tools(Default::default()).await {
        Ok(l) => l,
        Err(e) => return Err(McpIntegrationError::DiscoveryError(e.to_string())),
    };
    Ok((Arc::new(Mutex::new(McpClientType::SseClient(client))), tool_list.tools))
}


pub async fn get_mcp_streamable_http_tools<T>(url: T) -> Result<(McpClient, Vec<rmcp::model::Tool>), McpIntegrationError> where T: Into<String> {
    let transport = StreamableHttpClientTransport::from_uri(url.into());
    let client_info = ClientInfo {
        protocol_version: Default::default(),
        capabilities: ClientCapabilities::default(),
        client_info: Implementation {
            name: "test sse client".to_string(),
            version: "0.0.1".to_string(),
        },
    };
    let client = match client_info.serve(transport).await {
        Ok(c) =>c,
        Err(e) => return Err(McpIntegrationError::ConnectionError(e.to_string())),
    };

    let tool_list = match client.list_tools(Default::default()).await {
        Ok(l) => l,
        Err(e) => return Err(McpIntegrationError::DiscoveryError(e.to_string())),
    };
    Ok((Arc::new(Mutex::new(McpClientType::StreamableHttp(client))), tool_list.tools))
}


pub async fn get_mcp_stdio_tools<T>(full_command: T) -> Result<(McpClient, Vec<rmcp::model::Tool>), McpIntegrationError> where T: Into<String> {
    let full_command_string = full_command.into();
    let mut command_args = full_command_string.split(" ");
    let first = command_args.next();
    if first.is_none() {
        return Err(McpIntegrationError::ConnectionError("Invalid command.".into()));
    }
    let transport = match TokioChildProcess::new(Command::new(first.unwrap()).configure(
        |cmd| {
            for arg in command_args {
                cmd.arg(arg);
            }
        },
    )) {
        Ok(t) =>t,
        Err(e) => return Err(McpIntegrationError::ConnectionError(e.to_string())),
    };

    let client = match ().serve(transport)
        .await {
            Ok(c) =>c,
            Err(e) => return Err(McpIntegrationError::ConnectionError(e.to_string())),
        };

    let tool_list = match client.list_tools(Default::default()).await {
        Ok(l) => l,
        Err(e) => return Err(McpIntegrationError::DiscoveryError(e.to_string())),
    };
    Ok((Arc::new(Mutex::new(McpClientType::StdioClient(client))), tool_list.tools))
}