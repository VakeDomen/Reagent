use std::{collections::HashMap, error, sync::{Arc}};

use serde_json::Value;
use tokio::sync::Mutex;

use crate::{services::ollama::models::tool::{Function, FunctionArguments, Property, Tool}, ToolBuilder, ToolExecutionError};

use super::error::McpIntegrationError;
use rmcp::{model::{CallToolRequestParam, ClientCapabilities, ClientInfo, Implementation, JsonObject, Tool as McpTool}, schemars::schema::{InstanceType, Schema, SingleOrVec}, service::RunningService, transport::{SseClientTransport, StreamableHttpClientTransport}, Service, ServiceExt};
use crate::AsyncToolFn;

pub enum McpServerType {
    Sse,
    Io,
}

pub async fn get_mcp_tools<T>(server_url: T, mcp_server_type: McpServerType) -> Result<Vec<Tool>, McpIntegrationError> where T: Into<String> {
    let (mcp_client_arc, mcp_raw_tools) = get_mcp_sse_tools(server_url).await?;

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


pub type ArcMcpClient = Arc<Mutex<RunningService<rmcp::RoleClient, rmcp::model::InitializeRequestParam>>>;
pub async fn get_mcp_sse_tools<T>(url: T) -> Result<(ArcMcpClient, Vec<rmcp::model::Tool>), McpIntegrationError> where T: Into<String> {
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
    Ok((Arc::new(Mutex::new(client)), tool_list.tools))
}