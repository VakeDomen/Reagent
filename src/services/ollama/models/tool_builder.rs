use std::collections::HashMap;

use super::tool::{AsyncToolFn, Function, FunctionParameters, Property, Tool, ToolType};


#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolBuilderError {
    MissingFunctionName,
    MissingFunctionDescription,
    MissingExecutor,
}

impl std::fmt::Display for ToolBuilderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ToolBuilderError::MissingFunctionName => write!(f, "Function name is required."),
            ToolBuilderError::MissingFunctionDescription => write!(f, "Function description is required."),
            ToolBuilderError::MissingExecutor => write!(f, "Executor function is required for the tool."),
        }
    }
}

impl std::error::Error for ToolBuilderError {}


#[derive(Default)]
pub struct ToolBuilder {
    tool_type: Option<ToolType>,
    function_name: Option<String>,
    function_description: Option<String>,
    function_properties: HashMap<String, Property>,
    function_required: Vec<String>,
    executor: Option<AsyncToolFn>,
}

impl std::fmt::Debug for ToolBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolBuilder")
            .field("tool_type", &self.tool_type)
            .field("function_name", &self.function_name)
            .field("function_description", &self.function_description)
            .field("function_properties", &self.function_properties)
            .field("function_required", &self.function_required)
            .field("executor", &self.executor.as_ref().map(|_| "<async_fn>")) // Show placeholder if executor is Some
            .finish()
    }
}

impl ToolBuilder {
    /// Creates a new `ToolBuilder`.
    /// By default, `tool_type` will be `ToolType::Function` and
    /// `function_param_type` will be `"object"` if not explicitly set.
    pub fn new() -> Self {
        ToolBuilder {
            tool_type: Some(ToolType::Function),
            function_properties: HashMap::new(),
            function_required: Vec::new(),
            ..Default::default()
        }
    }

    /// Sets the type of the tool.
    /// Defaults to `ToolType::Function`.
    pub fn tool_type(mut self, tool_type: ToolType) -> Self {
        self.tool_type = Some(tool_type);
        self
    }

    /// Sets the name of the function for the tool. (Required)
    pub fn function_name(mut self, name: impl Into<String>) -> Self {
        self.function_name = Some(name.into());
        self
    }

    /// Sets the description of the function for the tool. (Required)
    pub fn function_description<T>(mut self, description: T) -> Self where T: Into<String> {
        self.function_description = Some(description.into());
        self
    }


    /// Adds a property to the function's parameters.
    ///
    /// # parameters
    /// * `name` - The name of the property.
    /// * `property_type` - The JSON schema type of the property (e.g., "string", "number", "boolean").
    /// * `description` - A description of what the property represents.
    pub fn add_property(
        mut self,
        name: impl Into<String>,
        property_type: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        self.function_properties.insert(
            name.into(),
            Property {
                property_type: property_type.into(),
                description: description.into(),
            },
        );
        self
    }

    /// Marks a property as required for the function.
    /// The property must have been previously added using `add_property`.
    pub fn add_required_property(mut self, name: impl Into<String>) -> Self {
        // It's good practice to ensure the property exists before adding it to required,
        // but for simplicity in the builder, we'll assume the user calls add_property first.
        // Alternatively, the build method could validate this.
        self.function_required.push(name.into());
        self
    }

    /// Sets the asynchronous executor function for the tool. (Required for building)
    pub fn executor(mut self, exec: AsyncToolFn) -> Self {
        self.executor = Some(exec);
        self
    }

    /// Consumes the builder and attempts to create a `Tool`.
    ///
    /// # Errors
    /// Returns a `ToolBuilderError` if required fields are missing.
    pub fn build(self) -> Result<Tool, ToolBuilderError> {
        let function_name = self.function_name.ok_or(ToolBuilderError::MissingFunctionName)?;
        let function_description = self.function_description.ok_or(ToolBuilderError::MissingFunctionDescription)?;
        let executor = self.executor.ok_or(ToolBuilderError::MissingExecutor)?; // Check for executor

        let parameters = FunctionParameters {
            param_type:"object".to_string(),
            properties: self.function_properties,
            required: self.function_required,
        };

        let function = Function {
            name: function_name,
            description: function_description,
            parameters: parameters,
        };

        Ok(Tool {
            tool_type: self.tool_type.unwrap_or(ToolType::Function),
            function,
            executor, 
        })
    }
}


// In your tool_builder.rs or a dedicated test module
#[cfg(test)]
mod tests {
    use super::*; // Imports ToolBuilder, ToolBuilderError
    use crate::services::ollama::models::tool::{Tool, ToolType, Function, Property, AsyncToolFn};
    use std::sync::Arc;
    use serde_json::Value;

    // A simple placeholder executor for testing definitions
    fn create_dummy_executor() -> AsyncToolFn {
        Arc::new(|_args: Value| {
            Box::pin(async { Ok("dummy execution".to_string()) })
        })
    }

    #[test]
    fn tool_builder_valid_tool() {
        let tool_result = ToolBuilder::new()
            .function_name("test_tool")
            .function_description("A tool for testing")
            .add_property("param1", "string", "A string parameter")
            .add_required_property("param1")
            .executor(create_dummy_executor())
            .build();

        assert!(tool_result.is_ok());
        let tool = tool_result.unwrap();
        assert_eq!(tool.function.name, "test_tool");
        assert_eq!(tool.function.parameters.properties.get("param1").unwrap().property_type, "string");
        assert!(tool.function.parameters.required.contains(&"param1".to_string()));
    }

    #[test]
    fn tool_builder_missing_name_fails() {
        let tool_result = ToolBuilder::new()
            .function_description("A tool missing a name")
            .executor(create_dummy_executor())
            .build();
        assert!(tool_result.is_err());
        assert_eq!(tool_result.unwrap_err(), ToolBuilderError::MissingFunctionName);
    }

    #[test]
    fn tool_builder_missing_description_fails() {
        let tool_result = ToolBuilder::new()
            .function_name("test_tool_no_desc")
            .executor(create_dummy_executor())
            .build();
        assert!(tool_result.is_err());
        // Assuming you have a MissingFunctionDescription error variant
        assert_eq!(tool_result.unwrap_err(), ToolBuilderError::MissingFunctionDescription);
    }

    #[test]
    fn tool_builder_missing_executor_fails() {
        let tool_result = ToolBuilder::new()
            .function_name("test_tool_no_exec")
            .function_description("A tool missing an executor")
            .build();
        assert!(tool_result.is_err());
        // Assuming you have a MissingExecutor error variant
        assert_eq!(tool_result.unwrap_err(), ToolBuilderError::MissingExecutor);
    }
}
