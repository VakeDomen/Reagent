
use std::error::Error;
use reagent::{init_default_tracing, models::{invocation::{invocation_handler::{Flow, FlowFuture}, invocation_util::invoke_without_tools}}, Agent, AgentBuilder, Message};

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
    println!("{:#?}", resp);

    Ok(())
}

fn custom_flow<'a>(agent: &'a mut Agent, prompt: String) -> FlowFuture<'a> {
    Box::pin(async move {
        agent.history.push(Message::user(prompt));
        let response = invoke_without_tools(agent).await?;
        Ok(response.message)
    })    
}