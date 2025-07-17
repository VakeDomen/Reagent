use std::error::Error;
use reagent::{init_default_tracing, models::{flow::invocation_flows::{Flow, FlowFuture}}, Agent, AgentBuilder, Message};


#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    init_default_tracing();
    
    let api_key = "secret-key-123".to_string();

    // closure must me defined inline
    let mut agent = AgentBuilder::default()
        .set_model("qwen3:4b")
        .set_flow(Flow::new_closure(move |_: &mut Agent, _: String| -> FlowFuture<'_> {
            let api_key_clone = api_key.clone(); 
            Box::pin(async move {
                Ok(Message::assistant(format!("The key is: {}", api_key_clone)))
            })
        }))
        .build()
        .await?;

    println!("{:#?}", agent.invoke_flow("What's the key?").await?);
    
    Ok(())
}