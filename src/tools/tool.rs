use std::{collections::HashMap, fmt, future::Future, pin::Pin, sync::Arc};

use serde::{Deserialize, Serialize};
use serde_json::Value;

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
    #[serde(default = "default_tool_call_type", skip_serializing_if = "is_default_tool_call_type")]
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