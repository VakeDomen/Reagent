
//! # Ollama Rust Client
//!
//! A Rust module to interact with the Ollama API, providing a main client,
//! message types, and support for function calling and structured outputs.

// Re-export key components for easier use
pub use client::OllamaClient;
pub use models::{
    ChatRequest, ChatResponse, GenerateRequest, GenerateResponse, Message, Role, Tool, ToolCall,
    ToolType, Function, FunctionParameters, Property,
};

/// ## Main Client (`client.rs`)
///
/// This module contains the `OllamaClient`, which is responsible for making
/// requests to the Ollama API. It uses `reqwest` for asynchronous HTTP
/// requests and `serde` for JSON serialization/deserialization.
///
/// ### Features:
/// - Asynchronous requests.
/// - Base URL configuration.
/// - Handles various Ollama endpoints (Generate, Chat, etc.).
/// - Supports streaming responses (though simplified in this example).
// pub mod client {
    
// }

// ---


/// ## Example Usage (`main.rs`)
///
/// This example demonstrates how to use the `OllamaClient` to send a chat
/// request with a function-calling tool.
///
/// ```rust,no_run
/// use ollama_client::{OllamaClient, ChatRequest, Message, Role, Tool, ToolType, Function, FunctionParameters, Property};
/// use std::collections::HashMap;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let client = OllamaClient::default();
///
///     // 1. Define a tool
///     let get_weather_tool = Tool {
///         tool_type: ToolType::Function,
///         function: Function {
///             name: "get_current_weather".to_string(),
///             description: "Get the current weather for a specific location".to_string(),
///             parameters: FunctionParameters {
///                 param_type: "object".to_string(),
///                 properties: {
///                     let mut props = HashMap::new();
///                     props.insert(
///                         "location".to_string(),
///                         Property {
///                             property_type: "string".to_string(),
///                             description: "The city and state, e.g., San Francisco, CA".to_string(),
///                         },
///                     );
///                     props
///                 },
///                 required: vec!["location".to_string()],
///             },
///         },
///     };
///
///     // 2. Create the chat request with a message and the tool
///     let request = ChatRequest {
///         base: ollama_client::models::BaseRequest {
///             model: "llama3.1".to_string(), // Use a model that supports tool calling
///             format: None,
///             options: None,
///             stream: Some(false),
///             keep_alive: Some("5m".to_string()),
///         },
///         messages: vec![
///             Message {
///                 role: Role::User,
///                 content: "What's the weather like in Boston?".to_string(),
///                 images: None,
///                 tool_calls: None,
///                 tool_call_id: None,
///             },
///         ],
///         tools: Some(vec![get_weather_tool]),
///     };
///
///     // 3. Send the request
///     let response = client.chat(request).await?;
///
///     // 4. Check for tool calls in the response
///     if let Some(tool_calls) = &response.message.tool_calls {
///         for tool_call in tool_calls {
///             println!("Model wants to call function: {}", tool_call.function.name);
///             println!("With arguments: {:?}", tool_call.function.arguments);
///
///             // --- Here you would: ---
///             // a. Execute the actual function (e.g., call a weather API).
///             // b. Construct a new `Message` with `Role::Tool`, the result,
///             //    and the `tool_call.id`.
///             // c. Send a new `ChatRequest` including the tool response message.
///         }
///     } else {
///         println!("Model response: {}", response.message.content);
///     }
///
///     // Example for structured (JSON) output
///     let json_request = ChatRequest {
///         base: ollama_client::models::BaseRequest {
///             model: "llama3.1".to_string(),
///             format: Some("json".to_string()), // Request JSON output
///             options: None,
///             stream: Some(false),
///             keep_alive: Some("5m".to_string()),
///         },
///         messages: vec![
///             Message {
///                 role: Role::System,
///                 content: "You are a helpful assistant. Please respond with JSON containing a 'city' and 'weather' key.".to_string(),
///                 images: None,
///                 tool_calls: None,
///                 tool_call_id: None,
///             },
///             Message {
///                 role: Role::User,
///                 content: "What's the weather in London?".to_string(),
///                 images: None,
///                 tool_calls: None,
///                 tool_call_id: None,
///             },
///         ],
///         tools: None,
///     };
///
///     let json_response = client.chat(json_request).await?;
///     println!("Structured JSON response: {}", json_response.message.content);
///
///     Ok(())
/// }
/// ```
///
/// ### Notes:
/// - **Error Handling**: The error handling is basic; a production-ready client
///   would have more robust error types and handling.
/// - **Streaming**: This example focuses on non-streaming requests for simplicity.
///   Implementing streaming would require handling `reqwest::Response::chunk` and
///   parsing line-delimited JSON.
/// - **Dependencies**: You would need to add `reqwest` (with the `json` feature),
///   `serde`, `serde_json`, and `tokio` (with the `macros` and `rt-multi-thread`
///   features) to your `Cargo.toml`.
/// - **Model Support**: Ensure you are using an Ollama model version that
///   supports function calling and/or structured output.
/// - **Structured Output**: For structured output, you simply set `format: Some("json".to_string())`.
///   You *must* also instruct the model in the prompt to output JSON. The Ollama API
///   doesn't currently support passing a specific JSON *schema* directly in the same
///   way as function calling parameters, but `format: "json"` ensures the output is
///   valid JSON. For more complex schema enforcement, you'd typically use function calling.