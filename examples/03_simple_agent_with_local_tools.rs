
use std::{error::Error, sync::Arc};
use reagent::{init_default_tracing, AgentBuilder, AsyncToolFn, ToolBuilder};
use serde_json::Value;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    init_default_tracing();

    let weather_exec: AsyncToolFn = {
        Arc::new(move |_model_args_json: Value| {
            Box::pin(async move {
                Ok(r#"
                {
                "type":"object",
                "properties":{
                    "windy":{"type":"boolean"},
                    "temperature":{"type":"integer"},
                    "description":{"type":"string"}
                },
                "required":["windy","temperature","description"]
                }
                "#.into())
            })
        })
    };

    let weather_tool = ToolBuilder::new()
        .function_name("get_current_weather")
        .function_description("Returns a weather forecast for a given location")
        .add_property("location", "string", "City name")
        .add_required_property("location")
        .executor(weather_exec)
        .build()?;

    let mut agent = AgentBuilder::default()
        .set_model("qwen3:0.6b")
        .set_system_prompt("You are a helpful assistant.")
        .add_tool(weather_tool)
        .build()
        .await?;

    let resp = agent.invoke_flow("Say hello").await?;
    println!("\n-> Agent: {}", resp.content.unwrap_or_default());

    let resp = agent.invoke_flow("What is the current weather in Koper?").await?;
    println!("\n-> Agent: {}", resp.content.unwrap_or_default());

    let resp = agent.invoke_flow("What do you remember?").await?;
    println!("\n-> Agent: {}", resp.content.unwrap_or_default());

    Ok(())
}
