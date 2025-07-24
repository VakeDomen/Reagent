pub mod invocation_flows;
pub mod flows;


#[cfg(test)]
mod tests {
    // tests/invoke_flow_integration.rs
    use std::{error::Error, sync::Arc};
    use tokio::sync::Mutex;
    use serde_json::Value;

    use crate::{
        models::notification::NotificationContent, AgentBuilder, AsyncToolFn, Notification, ToolBuilder, ToolExecutionError
    };

    #[tokio::test]
    async fn end_to_end_weather_tool_flow() -> Result<(), Box<dyn Error>> {

        let mut weather_agent = AgentBuilder::default()
            .set_model("qwen3:0.6b")
            .strip_thinking(true)
            .set_system_prompt("/no_think \nYou make up weather info in JSON. You always say it's snowing")
            .set_seed(1)
            .set_temperature(0.0)
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
            .await
            .unwrap();

        let weather_ref = Arc::new(Mutex::new(weather_agent.clone()));
        let weather_exec: AsyncToolFn = {
            let weather_ref = weather_ref.clone();
            Arc::new(move |args: Value| {
                let weather_ref = weather_ref.clone();
                Box::pin(async move {
                    let mut agent = weather_ref.lock().await;
                    let loc = args.get("location")
                        .and_then(Value::as_str)
                        .ok_or_else(|| ToolExecutionError::ArgumentParsingError("Missing 'location'".into()))?;
                    let prompt = format!("/no_think What is the weather in {}?", loc);
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
            .build()
            .unwrap();

        let (mut agent, mut notifications_rx) = AgentBuilder::default()
            .set_model("qwen3:0.6b")
            .set_seed(1)
            .set_temperature(0.0)
            .set_system_prompt("You are a helpful assistant that follows the instuctions given by the user.")
            .add_tool(weather_tool)
            .build_with_notification()
            .await
            .unwrap();

        let notify_handle = tokio::spawn(async move {
        let mut seen = Vec::new();
            while let Some(note) = notifications_rx.recv().await {
                seen.push(note.clone());

                if let NotificationContent::Done(_) = note.content {
                    break;
                }
            }
            seen
        });

        let say_test_message = agent.invoke_flow("In your response you must include the word \
        'test'. Is the weather in Koper good enough I can go out to test my new bike? Remember to say 'test'.")
            .await
            .unwrap();
        assert!(say_test_message.content.unwrap().to_lowercase().contains("test"));

        let weather = weather_agent.invoke_flow("What is the current weather in Koper?")
            .await
            .unwrap();
        let weather_json = weather.content.unwrap();
        let v: Value = serde_json::from_str(&weather_json).unwrap();

        assert!(v.get("windy").is_some());
        assert!(v.get("temperature").is_some());
        assert!(v.get("description").is_some());

        let recall = agent.invoke_flow("What instructions were given to you by the user?")
            .await
            .unwrap();
        let rec_text = recall.content.unwrap_or_default();


        assert!(rec_text.to_lowercase().contains("test"));


        // check recieved notifications

        let notifications = notify_handle.await?;
        // 5. Now, run your assertions on the `notifications` vector.
        assert!(!notifications.is_empty(), "Expected at least one notification");
        assert!(notifications.iter().any(|n| matches!(n.content, NotificationContent::PromptRequest(_))));
        assert!(notifications.iter().any(|n| matches!(n.content, NotificationContent::ToolCallRequest(_))));
        assert!(notifications.iter().any(|n| matches!(n.content, NotificationContent::ToolCallSuccessResult(_))));
        assert!(notifications.iter().any(|n| matches!(n.content, NotificationContent::PromptSuccessResult(_))));
        assert!(notifications.iter().any(|n| matches!(n.content, NotificationContent::Done(_))));

        Ok(())
    }

}
