use std::error::Error;
use reagent::{
    init_default_tracing, 
    flow_types::{Flow, FlowFuture}, 
    Agent, 
    AgentBuilder, 
    Message 
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    init_default_tracing();
    
    let api_key = "secret-key-123".to_string();

    // you can also define closures as custom invocation
    // flows for the agent, but the closure must be
    // defined inline
    let mut agent = AgentBuilder::default()
        .set_model("qwen3:0.6b")

        // use Flow::new_closure to define the closture
        .set_flow(Flow::new_closure(move |_: &mut Agent, _prompt: String| -> FlowFuture<'_> {
            let api_key_clone = api_key.clone(); 
            Box::pin(async move {
                // dummy case, we just return the api the response
                // insert your logic here
                Ok(Message::assistant(format!("The key is: {api_key_clone}")))
            })
        }))
        .build()
        .await?;

    println!("{:#?}", agent.invoke_flow("What's the key?").await?);
    
    Ok(())
}