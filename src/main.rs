
use std::sync::Arc;

use agent::{models::AgentBuilder, AsyncToolFn, ToolBuilder, ToolExecutionError};
use anyhow::Result;
use serde_json::Value;

#[tokio::main]
async fn main() -> Result<()> {

    let my_executor: AsyncToolFn = Arc::new(|args: Value| {
        Box::pin(async move {
            println!("Executing with args: {:?}", args);
            if let Some(location) = args.get("location").and_then(|v| v.as_str()) {
                Ok(format!(
                    "The weather in {} is sunny and 25°C.",
                    location
                ))
            } else {
                Err(ToolExecutionError::ArgumentParsingError(
                    "Missing 'location' argument".to_string(),
                ))
            }
        })
    });

    let get_weather_tool = ToolBuilder::new()
        .function_name("get_current_weather")
        .function_description("Get the current weather for a specific location")
        .add_property("location", "string", "The city and state, e.g., San Francisco, CA")
        .add_required_property("location")
        .add_property("unit", "string", "Temperature unit (celsius or fahrenheit)") // Optional parameter
        .executor(my_executor)
        .build()?; // The '?' will propagate ToolBuilderError if it occurs.
 

    let mut agent = AgentBuilder::default()
        .set_model("mistral-nemo")
        .set_system_prompt("You are a helpful agent")
        .set_ollama_endpoint("http://hivecore.famnit.upr.si")
        .set_ollama_port(6666)
        .add_tool(get_weather_tool)
        .build();

    let resp = agent.invoke("Can you say 'Yeah'").await;
    println!("Agent Resp: {:#?}", resp);


    let resp = agent.invoke("What is the current weather in Ljubljana?").await;
    println!("Agent Resp: {:#?}", resp);

    println!("AGENT: {:#?}", agent);

    Ok(())
}
