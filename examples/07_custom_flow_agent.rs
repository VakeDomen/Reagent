
use std::error::Error;
use reagent::{
    init_default_tracing, 
    models::agents::flow::invocation_flows::{Flow, FlowFuture}, Agent, AgentBuilder, Message,
    util::invocations::invoke_without_tools
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    init_default_tracing();
    
    let mut agent = AgentBuilder::default()
        .set_model("qwen3:4b")
        .set_system_prompt("You are a helpful, assistant.")
        .set_flow(Flow::Custom(custom_flow))
        .build()
        .await?;

    let resp = agent.invoke_flow("What is the meaning of lige?").await?;
    println!("{resp:#?}");

    Ok(())
}

fn custom_flow<'a>(agent: &'a mut Agent, prompt: String) -> FlowFuture<'a> {
    Box::pin(async move {
        agent.history.push(Message::user(prompt));
        let response = invoke_without_tools(agent).await?;
        Ok(response.message)
    })    
}