use std::{collections::HashMap, fmt, future::Future, pin::Pin, sync::Arc};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{Agent, Message, NotificationHandler};

use super::errors::ToolExecutionError;

/// Defines the type of tool available. Currently, only 'function' is supported.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ToolType {
    Function,
}

/// Signature for an asynchronous tool executor function.
///
/// Accepts a JSON [`Value`] of arguments and produces a `String` result
/// or a [`ToolExecutionError`] if execution fails.
pub type AsyncToolFn = Arc<
    dyn Fn(Value) -> Pin<Box<dyn Future<Output = Result<String, ToolExecutionError>> + Send>>
        + Send
        + Sync,
>;

/// A placeholder function for deserialization.
/// panic if called, indicating a logic error where a tool was
/// deserialized but not properly re-initialized.
fn default_executor() -> AsyncToolFn {
    Arc::new(|_| {
        Box::pin(async {
            panic!("Called a default, non-functional tool executor. The tool was not rehydrated after deserialization.");
        })
    })
}

/// Defines a tool (function) that the model can call.
#[derive(Serialize, Clone, Deserialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tool_type: ToolType,
    pub function: Function,
    #[serde(skip, default = "default_executor")]
    pub executor: AsyncToolFn,
}

impl fmt::Debug for Tool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tool")
            .field("tool_type", &self.tool_type)
            .field("function", &self.function)
            .field("executor", &"<async_fn>") // Placeholder for the executor
            .finish()
    }
}

impl Tool {
    /// Convenience method to execute the tool
    pub async fn execute(&self, args: Value) -> Result<String, ToolExecutionError> {
        (self.executor)(args).await
    }

    /// Gets the name of the tool from its function definition.
    pub fn name(&self) -> &str {
        &self.function.name
    }
}

/// Defines a function, its description, and its arguments.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Function {
    pub name: String,
    pub description: String,
    pub parameters: FunctionParameters,
}

/// Defines the arguments for a function using a JSON schema-like structure.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FunctionParameters {
    #[serde(rename = "type")]
    pub param_type: String,
    pub properties: HashMap<String, Property>,
    pub required: Vec<String>,
}

/// Defines a single property within function arguments.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Property {
    #[serde(rename = "type")]
    pub property_type: String,
    pub description: String,
}

/// Represents a tool call requested by the model.
///
/// Tool calls reference a function name and include JSON arguments.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ToolCall {
    /// Optional identifier for the tool call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// The type of the tool (defaults to [`ToolType::Function`]).
    ///
    /// Some providers omit this field, so a default is supplied.
    #[serde(
        default = "default_tool_call_type",
        skip_serializing_if = "is_default_tool_call_type"
    )]
    #[serde(rename = "type")]
    pub tool_type: ToolType,
    /// Function being called.
    pub function: ToolCallFunction,
}

// Helper function to provide a default ToolType if it's missing in the JSON
// This is used if your Ollama version consistently omits the "type" field in the tool_call object.
fn default_tool_call_type() -> ToolType {
    ToolType::Function // Default to function
}

// Helper for skip_serializing_if to avoid serializing if it's the default
// This is more relevant if you were to serialize this struct often and wanted to omit default values.
#[allow(clippy::trivially_copy_pass_by_ref)]
fn is_default_tool_call_type(tool_type: &ToolType) -> bool {
    *tool_type == default_tool_call_type()
}

/// Contains the name and arguments for a function call.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ToolCallFunction {
    pub name: String,
    pub arguments: serde_json::Value,
}

/// Execute a batch of tool calls and return their messages.
///
/// For each [`ToolCall`] in the input slice:
/// - Looks up the corresponding tool in the agent’s registry.
/// - Executes it asynchronously with the provided arguments.
/// - Emits notifications for request, success, or error.
/// - Produces a [`Message`] representing the tool output.
///
/// Returns a `Vec<Message>` containing all tool responses (including
/// error placeholders when a tool cannot be found or fails).
pub async fn call_tools(agent: &Agent, tool_calls: &[ToolCall]) -> Vec<Message> {
    let mut results = Vec::new();

    let Some(avail) = &agent.tools else {
        tracing::error!("No avalible tools specified");

        agent
            .notify_tool_error("Agent called tools, but no tools avalible to the model".into())
            .await;

        results.push(Message::tool(
            "If you want to use a tool specify the name of the available tool.",
            "Tool".to_string(),
        ));

        return results;
    };

    for call in tool_calls {
        tracing::info!(
            target: "tool",
            tool = %call.function.name,
            id   = ?call.id,
            args = ?call.function.arguments,
            "executing tool call",
        );

        // try to find the tool
        let Some(tool) = avail.iter().find(|t| t.function.name == call.function.name) else {
            tracing::error!("No corresponding tool found.");
            let msg = format!("Could not find tool: {}", call.function.name);
            agent.notify_tool_error(msg.clone()).await;
            results.push(Message::tool(msg, "0".to_string()));
            continue;
        };

        agent.notify_tool_request(call.clone()).await;

        match tool.execute(call.function.arguments.clone()).await {
            Ok(output) => {
                agent.notify_tool_success(output.clone()).await;
                results.push(Message::tool(
                    output,
                    call.id.clone().unwrap_or(call.function.name.clone()),
                ));
            }
            Err(e) => {
                agent.notify_tool_error(e.to_string()).await;
                let msg = format!("Error executing tool {}: {}", call.function.name, e);
                results.push(Message::tool(
                    msg,
                    call.id.clone().unwrap_or(call.function.name.clone()),
                ));
            }
        }
    }

    results
}
