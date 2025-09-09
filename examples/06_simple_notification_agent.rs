
use std::{collections::HashMap, error::Error};
use reagent_rs::{AgentBuilder, NotificationContent};
use serde_json::to_value;


#[derive(serde::Serialize)]
struct MyCustomNotification {
    message: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {

    let (mut agent, mut notification_reciever) = AgentBuilder::default()
        .set_model("qwen3:0.6b")
        .set_system_prompt("You are a helpful, assistant.")
        // if stream is set to true, the agent
        // will also return Token notifications
        .set_stream(true)
        // when you use `build_with_notification` 
        // instead of `build`, the AgentBuilder
        // also returns a `Reviecer<Notification>`
        // channel that you can use to recieve 
        // notifications from the agent
        .build_with_notification()
        .await?;

    // spawn an async task that counts the number of notification
    // types recieved from the agent
    let handle = tokio::spawn(async move {
        println!("Counting...");
        // store counts
        let mut counts: HashMap<&'static str, usize> = HashMap::new();

        while let Some(msg) = notification_reciever.recv().await {
            // map notification type
            let type_name = match msg.content {
                NotificationContent::Done(_,_)=>{print!("{:#?}",msg);"Done"},
                NotificationContent::PromptRequest(_)=>"PromptRequest",
                NotificationContent::PromptSuccessResult(_)=>"PromptSuccessResult",
                NotificationContent::PromptErrorResult(_)=>"PromptErrorResult",
                NotificationContent::ToolCallRequest(_)=>"ToolCallRequest",
                NotificationContent::ToolCallSuccessResult(_)=>"ToolCallSuccessResult",
                NotificationContent::ToolCallErrorResult(_)=>"ToolCallErrorResult",
                NotificationContent::McpToolNotification(_)=>"McpToolNotification",
                NotificationContent::Token(t)=>{print!("{}",t.value);"Token"},
                NotificationContent::Custom(value) =>{ print!("{}",value);"Custom"},
            };

            // Increment the count for that type
            *counts.entry(type_name).or_default() += 1;

        }

        // This block runs after the channel closes (i.e., the agent is dropped).
        println!("\n--- Notification Summary ---");
        if counts.is_empty() {
            println!("No notifications were received.");
        } else {
            for (name, count) in counts {
                println!("{name}: {count}");
            }
        }
        println!("--------------------------\n");
    });

    let _resp = agent.invoke_flow("Say hello").await?;
    let _resp = agent.invoke_flow("What is the current weather in Koper?").await?;
    let _resp = agent.invoke_flow("What do you remember?").await?;



    let my_notification = MyCustomNotification {
        message: "This is a custom notification".to_string(),
    };


    agent.notify(NotificationContent::Custom(
        to_value(&my_notification).unwrap()
    )).await;

    // dropping agent so the comm channel closes and the tokio thread desplays 
    // the counts of notifications
    drop(agent);

    handle.await?;

    Ok(())
}
