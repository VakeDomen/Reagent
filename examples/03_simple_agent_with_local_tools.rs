
use std::{error::Error, sync::Arc};
use reagent_rs::{init_default_tracing, AgentBuilder, AsyncToolFn, ToolBuilder, ToolExecutionError};
use serde_json::Value;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    init_default_tracing();

    // what do you want to happen when the model calls the function?
    // Arcs and Boxes are needed to make the tool async and clonable
    let weather_exec: AsyncToolFn = {
        Arc::new(move |_model_args_json: Value| {
            Box::pin(async move {
                // dummy functionality jsut returning a fixed JSON
                // put your logic here
                Ok(r#"
                {
                    "windy": false,
                    "temperature": 18,
                    "description": "Partly cloudy"
                }
                "#.into()) // return Ok(String) or Err(AgentError)
            })
        })
    };

    // consturct the tool
    let weather_tool = ToolBuilder::new()
        .function_name("get_current_weather")
        .function_description("Returns a weather forecast for a given location")
        .add_required_property("location", "string", "City name")
        // closure that triggers on tool use
        .executor(weather_exec)
        .build()?;


    let echo_tool = ToolBuilder::new()
        .function_name("Echo")
        .function_description("Echos bach the input")
        .add_required_property("text", "string", "Text to echo")
        .executor_fn(echo)
        .build()?;

    let mut agent = AgentBuilder::default()
        .set_model("qwen3:0.6b")
        .set_system_prompt("You are a helpful assistant.")
        // add the tool to agent
        .add_tool(weather_tool)
        .add_tool(echo_tool)
        .build()
        .await?;

    let resp = agent.invoke_flow("Say hello").await?;
    println!("Agent: {}", resp.content.unwrap_or_default());

    let resp = agent.invoke_flow("What is the current weather in Koper?").await?;
    println!("Agent: {}", resp.content.unwrap_or_default());

    Ok(())
}


async fn echo(input: Value) -> Result<String, ToolExecutionError> {
    Ok(input.to_string())
}