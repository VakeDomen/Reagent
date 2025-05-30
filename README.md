# Simple Rust AI Agent with Ollama


```rust

use anyhow::Result;
use std::sync::Arc;
use tokio::sync::Mutex; 
use agent::{AgentBuilder, AsyncToolFn, McpServerType, ToolBuilder, ToolExecutionError, Value};


#[tokio::main] 
async fn main() -> Result<()> {
    // --- Define a specialized "Weather Agent" that will act as a tool ---
    // This agent is designed to make up weather information and respond in a specific JSON format.
    // It demonstrates how one agent can be encapsulated and used as a capability by another.
    let weather_agent = AgentBuilder::default()
        // Specifies the Ollama model for this sub-agent.
        .set_model("qwen3:30b") 
        // The system prompt directs the sub-agent's behavior:
        // - "/no_think": A common convention to tell the LLM to respond directly without
        //   outputting its internal "thought process" or chain-of-thought markers like <think>...</think>.
        // - The rest of the prompt defines its persona and task.
        .set_system_prompt("/no_think \nYou make up weather when given a location. Act like you know. ")
        // `set_response_format` tells the sub-agent's LLM to structure its output as JSON
        // conforming to the provided JSON schema. This makes the sub-agent's output predictable
        // and easily parsable by the tool executor.
        .set_response_format(r#"{
                "type": "object",
                "properties": {
                    "windy": {
                        "type": "boolean"
                    },
                    "temperature": {
                        "type": "integer"
                    },
                    "overlook": {
                        "type": "string"
                    }
                },
                "required": [
                    "windy",
                    "temperature",
                    "overlook"
                ]
            }"#
        )
        .build() // Consumes the builder and creates the Agent.
        .await?; // `.await` because AgentBuilder::build() is async (e.g., for MCP tool loading).
                 // `?` propagates any error from the build process.

    // Wrap the `weather_agent` in an `Arc<Mutex<T>>`.
    // - `Arc`: Allows the agent to be safely shared across multiple async tasks or calls if needed.
    //          The executor closure will capture this Arc.
    // - `Mutex`: Ensures that if the agent's `invoke` method requires mutable access (`&mut self`),
    //            only one execution of the tool can access it at a time, preventing data races.
    let weather_agent_ref = Arc::new(Mutex::new(weather_agent));

    // Define the executor function for the "get_current_weather" tool.
    // `AsyncToolFn` is a type alias for a complex function signature suitable for async tool execution.
    // It's an `Arc`'d, `Send + Sync` trait object representing a function that:
    //   - Takes `serde_json::Value` (arguments from the main LLM).
    //   - Returns a Pinned, Boxed, Sendable Future, which resolves to `Result<String, ToolExecutionError>`.
    let weather_agent_tool_executor: AsyncToolFn = Arc::new(move |args: Value| {
        // The `move` keyword on this outer closure is crucial. It ensures that
        // `weather_agent_ref` (the Arc pointing to the sub-agent) is moved into and owned by this closure.
        // This makes the closure (and everything it owns) 'static, satisfying lifetime requirements
        // for storing it as an AsyncToolFn.

        // For each call to this tool, we clone the Arc.
        // This allows the returned Future to own its reference to the shared sub-agent.
        let agent_arc_for_this_call = Arc::clone(&weather_agent_ref);

        // `Box::pin` creates a `Pin<Box<dyn Future>>`.
        // - `Box`: Allocates the Future on the heap, giving it a stable address and known size (for type erasure).
        // - `Pin`: Prevents the Future from being moved in memory after it has been polled,
        //          which is necessary for safety with self-referential async state.
        // - `async move`: The block creates the Future. `move` ensures variables from the surrounding
        //                 scope (like `agent_arc_for_this_call` and `args`) are moved into the Future.
        Box::pin(async move {
            println!("[Tool Executor] Weather tool called with args: {:?}", args);

            // Lock the mutex to get mutable access to the weather_agent.
            // The lock is held until `agent_guard` goes out of scope.
            let mut agent_guard = agent_arc_for_this_call.lock().await;

            if let Some(location) = args.get("location").and_then(|v| v.as_str()) {
                // Construct a specific prompt for the weather sub-agent.
                let prompt = format!("/no_think What is the weather at: {}", location);

                // Invoke the sub-agent.
                match agent_guard.invoke(prompt).await {
                    Ok(message_from_agent) => {
                        // The sub-agent was prompted with "/no_think" and a JSON response_format.
                        // Its `message_from_agent.content` should ideally be the JSON string.
                        match message_from_agent.content {
                            Some(text_content) => {
                                // This parsing for `</think>` is a defensive measure.
                                // Given the "/no_think" prompt to the sub-agent, this tag might not appear.
                                // If it can, this strips any "thought process" that might have slipped through.
                                let tag = "</think>";
                                if let Some((_before_tag, after_tag)) = text_content.rsplit_once(tag) {
                                    Ok(after_tag.trim().to_string())
                                } else {
                                    // If no tag, the content (which should be JSON) is used as is.
                                    Ok(text_content.trim().to_string())
                                }
                            }
                            None => {
                                // Fallback if the sub-agent somehow returned no content.
                                Ok("Weather agent provided no specific content.".to_string())
                            }
                        }
                    }
                    Err(agent_error) => {
                        // If the sub-agent invocation fails, wrap its error into a ToolExecutionError.
                        Err(ToolExecutionError::ExecutionFailed(format!(
                            "Weather sub-agent invocation failed: {}",
                            agent_error // Assumes agent_error implements Display
                        )))
                    }
                }
            } else {
                // If the required "location" argument was missing for the tool.
                Err(ToolExecutionError::ArgumentParsingError(
                    "Missing 'location' argument".to_string(),
                ))
            }
        })
    });


    // --- Define the "get_current_weather" tool for the main agent ---
    // `ToolBuilder` is used to define the tool's interface (name, description, parameters)
    // that the main LLM will see and understand how to call.
    let get_weather_tool = ToolBuilder::new()
        .function_name("get_current_weather")
        .function_description("Get the current weather for a specific location. This will be a made-up forecast.")
        .add_property("location", "string", "The city and state, e.g., San Francisco, CA")
        .add_required_property("location") // Marks "location" as a mandatory parameter.
        .add_property("unit", "string", "Optional. Temperature unit (celsius or fahrenheit). Currently ignored by the tool.")
        .executor(weather_agent_tool_executor) // Attaches the actual Rust async code to be run.
        .build()?; // Builds the Tool instance, potentially returning an error if definition is invalid.
 

    // --- Define and build the main "Helpful Agent" ---
    let mut agent = AgentBuilder::default()
        .set_model("qwen3:30b") // Model for the main agent.
        .set_system_prompt("/no_think \nYou are a helpful agent that can use tools to answer questions.")
        .add_tool(get_weather_tool) // Statically adds the weather tool we defined above.
        // Dynamically adds tools from MCP (Model Context Protocol) servers.
        // The agent will connect to these servers on build, discover available tools,
        // and create executors for them automatically.
        .add_mcp_server(McpServerType::stdio( // For tools exposed via a local command-line process.
            // "npx -y @modelcontextprotocol/server-everything" is an example command
            // that starts an MCP server providing various tools.
            // Ensure `npx` is in your PATH and the package can be downloaded if running this.
            // Note: For production, you'd typically use a more stable way to run MCP servers.
            "npx -y @modelcontextprotocol/server-everything"
        ))
        .add_mcp_server(McpServerType::streamable_http( // For tools exposed over HTTP/SSE.
            "http://localhost:8000/mcp" // Example URL for an MCP server.
        ))
        .build() // Builds the main agent. This is async due to MCP tool discovery.
        .await?;

    // --- Interact with the main agent ---
    // Example 1: A simple interaction that doesn't require tools.
    let resp1 = agent.invoke("Can you say 'Yeah'").await?;
    println!("Agent Resp 1 (Yeah): {:#?}", resp1.content.unwrap_or_default());

    // Example 2: Invoking the "get_current_weather" tool.
    // The LLM should identify that this question can be answered by the tool,
    // then call it, get the (made-up JSON) result from our sub-agent,
    // and finally formulate a natural language answer.
    let resp2 = agent.invoke("What is the current weather in Ljubljana?").await?;
    println!("Agent Resp 2 (Ljubljana Weather): {:#?}", resp2.content.unwrap_or_default());

    // Example 3: Invoking a tool potentially discovered from an MCP server
    // (e.g., the "incrementer" tool from @modelcontextprotocol/server-everything).
    let resp3 = agent.invoke("can you increment 3 times and show the final value?").await?;
    println!("Agent Resp 3 (Incrementer Tool): {:#?}", resp3.content.unwrap_or_default());

    // You can inspect the agent's history if needed:
    // println!("Final Agent History: {:#?}", agent.get_history());

    Ok(())
}

```