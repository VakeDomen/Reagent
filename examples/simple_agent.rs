
use std::{error::Error, sync::Arc};
use tokio::sync::Mutex;
use reagent::{AgentBuilder, AsyncToolFn, McpServerType, ToolBuilder, ToolExecutionError};
use serde_json::Value;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let weather_agent = AgentBuilder::default()
        .set_model("granite3-moe")
        .set_system_prompt("/no_think \nYou make up weather info in JSON. You always say it's sowing")
        .set_response_format(
            r#"
            {
              "type":"object",
              "properties":{
                "windy":{"type":"boolean"},
                "temperature":{"type":"integer"},
                "description":{"type":"string"}
              },
              "required":["windy","temperature","description"]
            }
            "#,
        )
        .build()
        .await?;

    let weather_ref = Arc::new(Mutex::new(weather_agent));
    let weather_exec: AsyncToolFn = {
        let weather_ref = weather_ref.clone();
        Arc::new(move |args: Value| {
            let weather_ref = weather_ref.clone();
            Box::pin(async move {
                let mut agent = weather_ref.lock().await;
                
                let loc = args.get("location")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| ToolExecutionError::ArgumentParsingError("Missing 'location' argument".into()))?;

                let prompt = format!("/no_think What is the weather in {}?", loc);

                let resp = agent.invoke(prompt)
                    .await
                    .map_err(|e| ToolExecutionError::ExecutionFailed(e.to_string()))?;
                Ok(resp.content.unwrap_or_default())
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
        .set_model("qwen3:30b")
        .set_system_prompt("You are a helpful, assistant.")
        .add_mcp_server(McpServerType::stdio("npx -y @modelcontextprotocol/server-memory"))
        .add_tool(weather_tool)
        .build()
        .await?;

    let resp = agent.invoke("Say hello").await?;
    println!("\n-> Agent: {}", resp.content.unwrap_or_default());

    let resp = agent.invoke("What is the current weather in Koper?").await?;
    println!("\n-> Agent: {}", resp.content.unwrap_or_default());

    let resp = agent.invoke("What do you remember?").await?;
    println!("\n-> Agent: {}", resp.content.unwrap_or_default());

    Ok(())
}
