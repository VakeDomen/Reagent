
use std::error::Error;
use reagent::{init_default_tracing, prebuilds::stateless::StatelessPrebuild, NotificationContent};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    init_default_tracing();
    

    let (mut agent, mut notification_reciever) = StatelessPrebuild::reply()
        .set_model("qwen3:0.6b")
        .set_stream(true)
        .build_with_notification()
        .await?;

    tokio::spawn(async move {
        while let Some(msg) = notification_reciever.recv().await {
            match msg.content {
                NotificationContent::Token(t)=>print!("{}", t.value),
                _ => ()
            }
        }
    });

    let _resp = agent.invoke_flow("Say hello").await?;
    let _resp = agent.invoke_flow("What is the current weather in Koper?").await?;
    let _resp = agent.invoke_flow("What do you remember?").await?;

    Ok(())
}
