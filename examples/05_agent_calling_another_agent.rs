
use std::{error::Error, sync::Arc};
use tokio::sync::Mutex;
use reagent_rs::{init_default_tracing, AgentBuilder, AsyncToolFn, ToolBuilder, ToolExecutionError};
use serde_json::Value;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    init_default_tracing();

    // agent A will be able to call agent B via a tool call
    // we just pack invoking agent B into a tool and pass it to 
    // the agent A on construction

    // in this case the agent B (tool) will make-up some weather data


    // another agent inside a local tool
    let weather_agent_b = AgentBuilder::default()
        .set_model("qwen3:0.6b")
        .set_system_prompt("You make up weather info in JSON. You always say it's sowing")
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

    
    // pass the model ref into the closure, so when the funcion is called adn
    // the closure triggers the model is invoked
    let weather_ref = Arc::new(Mutex::new(weather_agent_b));
    let weather_exec: AsyncToolFn = {
        let weather_ref = weather_ref.clone();
        Arc::new(move |args: Value| {
            let weather_ref = weather_ref.clone();
            Box::pin(async move {
                let mut agent = weather_ref.lock().await;
                // get "location" parameter from the args (JSON value)
                let loc = args.get("location")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| ToolExecutionError::ArgumentParsingError("Missing 'location' argument".into()))?;

                // create prompt for our agent B
                let prompt = format!("/no_think What is the weather in {loc}?");

                // invoke it
                let resp = agent.invoke_flow(prompt)
                    .await
                    .map_err(|e| ToolExecutionError::ExecutionFailed(e.to_string()))?;

                // return the response
                Ok(resp.content.unwrap_or_default())
            })
        })
    };

    // build the tool
    let agent_b_tool = ToolBuilder::new()
        .function_name("get_current_weather")
        .function_description("Returns a weather forecast for a given location")
        .add_required_property("location", "string", "City name")
        .executor(weather_exec)
        .build()?;

    // build the agent
    let mut agent = AgentBuilder::default()
        .set_model("qwen3:0.6b")
        .set_system_prompt("You are a helpful, assistant.")
        .add_tool(agent_b_tool)
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
