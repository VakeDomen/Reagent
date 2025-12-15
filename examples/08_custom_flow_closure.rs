use reagent_rs::{Agent, AgentBuilder, FlowFuture, Message};
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    reagent_rs::observability::init_default_tracing();

    let api_key = "secret-key-123".to_string();

    // you can also define closures as custom invocation
    // flows for the agent
    let mut agent = AgentBuilder::default()
        .set_model("qwen3:0.6b")
        .set_flow(move |_: &mut Agent, _prompt: String| -> FlowFuture<'_> {
            let api_key_clone = api_key.clone();
            Box::pin(async move {
                // dummy case, we just return the api the response
                // insert your logic here
                Ok(Message::assistant(format!("The key is: {api_key_clone}")))
            })
        })
        .build()
        .await?;

    println!("{:#?}", agent.invoke_flow("What's the key?").await?);

    Ok(())
}
