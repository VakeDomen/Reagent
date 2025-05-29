
use std::sync::Arc;

use agent::{json, models::AgentBuilder, AsyncToolFn, ToolBuilder, ToolExecutionError, Value};
use anyhow::Result;
use tokio::sync::Mutex;

#[tokio::main]
async fn main() -> Result<()> {
    let weather_agent = AgentBuilder::default()
        .set_model("qwen3:30b")
        .set_system_prompt("/no_think \nYou make up weather when given a location. Act like you know. ")
        .set_ollama_endpoint("http://hivecore.famnit.upr.si")
        .set_ollama_port(6666)
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
        .build()?;

    let weather_agent_ref = Arc::new(Mutex::new(weather_agent));

    let weather_agent_tool_executor: AsyncToolFn = Arc::new(move |args: Value| {
        Box::pin({
        let agent_arc_for_this_call = Arc::clone(&weather_agent_ref);
        async move {
            println!("Executing with args: {:?}", args);
            let mut agent = agent_arc_for_this_call.lock().await;

            if let Some(location) = args.get("location").and_then(|v| v.as_str()) {
                let prompt = format!("/no_think What is the weather at: {}", location);

                match agent.invoke(format!("What is the weather at: {}", location)).await {
                    Ok(message_from_agent) => {
                        match message_from_agent.content {
                            Some(text_content) => {
                                let tag = "</think>";
                                if let Some((_before_tag, after_tag)) = text_content.rsplit_once(tag) {
                                    Ok(after_tag.trim().to_string())
                                } else {
                                    Ok(text_content.trim().to_string())
                                }
                            }
                            None => {
                                // The content was None to begin with.
                                Ok("Weather agent provided no specific content.".to_string())
                            }
                        }
                    }
                    Err(agent_error) => {
                        // Convert your agent's specific error into ToolExecutionError
                        Err(ToolExecutionError::ExecutionFailed(format!(
                            "Weather agent invocation failed: {}",
                            agent_error // Assuming YourAgentError implements Display
                        )))
                    }
                }
            } else {
                Err(ToolExecutionError::ArgumentParsingError(
                    "Missing 'location' argument".to_string(),
                ))
            }
        }
        })
    });


    let get_weather_tool = ToolBuilder::new()
        .function_name("get_current_weather")
        .function_description("Get the current weather for a specific location")
        .add_property("location", "string", "The city and state, e.g., San Francisco, CA")
        .add_required_property("location")
        .add_property("unit", "string", "Temperature unit (celsius or fahrenheit)") 
        .executor(weather_agent_tool_executor)
        .build()?;
 

    let mut agent = AgentBuilder::default()
        .set_model("qwen3:30b")
        .set_system_prompt("/no_think \nYou are a helpful agent")
        .set_ollama_endpoint("http://hivecore.famnit.upr.si")
        .set_ollama_port(6666)
        .add_tool(get_weather_tool)
        .build()?;

    let resp = agent.invoke("Can you say 'Yeah'").await;
    // println!("Agent Resp: {:#?}", resp?.content);


    let resp = agent.invoke("What is the current weather in Ljubljana?").await;
    // println!("Agent Resp: {:#?}", resp?.content);

    println!("Agen: {:#?}", agent);


    Ok(())
}
