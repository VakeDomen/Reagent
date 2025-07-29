
use std::{error::Error, sync::Arc};
use tokio::sync::Mutex;
use reagent::{init_default_tracing, AgentBuilder, AsyncToolFn, NotificationContent, ToolBuilder, ToolExecutionError};
use serde_json::Value;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    init_default_tracing();
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

                let prompt = format!("/no_think What is the weather in {loc}?");

                let resp = agent.invoke_flow(prompt)
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

    let (mut agent, mut notification_reciever) = AgentBuilder::default()
        .set_model("qwen3:30b")
        .set_system_prompt("You are a helpful, assistant.")
        .add_tool(weather_tool)
        .build_with_notification()
        .await?;

    tokio::spawn(async move {
        while let Some(msg) = notification_reciever.recv().await {
            match msg.content {
                NotificationContent::ToolCallRequest(notification)=>println!("Recieved tool call reuqest notification: {notification:#?}"),
                NotificationContent::ToolCallSuccessResult(notification)=>println!("Recieved tool call Success notification: {notification:#?}"),
                NotificationContent::ToolCallErrorResult(notification)=>println!("Recieved tool call Error notification: {notification:#?}"),
                NotificationContent::Done(success, _) => println!("Done with generation: {success}"),
                _ => ()
            }
  
        }
    });

    let _resp = agent.invoke_flow("Say hello").await?;
    let _resp = agent.invoke_flow("What is the current weather in Koper?").await?;
    let _resp = agent.invoke_flow("What do you remember?").await?;

    Ok(())
}
