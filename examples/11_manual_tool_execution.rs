
use std::{error::Error, sync::Arc};
use reagent::{init_default_tracing, AgentBuilder, AsyncToolFn, ToolBuilder};
use serde_json::{json, Value};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    init_default_tracing();

    let weather_exec: AsyncToolFn = {
        Arc::new(move |_model_args_json: Value| {
            Box::pin(async move {
                Ok(r#"
                {
                    "windy": false,
                    "temperature": 18,
                    "description": "Partly cloudy"
                }
                "#.into())
            })
        })
    };

    // consturct the tool
    let weather_tool = ToolBuilder::new()
        .function_name("get_current_weather")
        .function_description("Returns a weather forecast for a given location")
        .add_required_property("location", "string", "City name")
        .executor(weather_exec)
        .build()?;


    // execute the tool
    let tool_response = weather_tool.execute(json!({
        "location": "Koper",
    })).await?;

    println!("Direct tool call response: {tool_response:#?}");



    // if you only have a ref to agent and the tool is constructed elsewhere 
    // (also maybe mcp tools)
    // you can extract the tools from agent with get_tool_ref_by_name
    let agent = AgentBuilder::default()
        .set_model("qwen3:0.6b")
        .set_system_prompt("You are a helpful assistant.")
        .add_tool(weather_tool)
        .build()
        .await?;

    
    let Some(tool) = agent.get_tool_ref_by_name("get_current_weather") else {
        panic!("No tool with that name found!");
    };

    let tool_response = tool.execute(json!({
        "location": "Koper",
    })).await?;

    println!("Extracted tool call response: {tool_response:#?}");

    Ok(())
}
