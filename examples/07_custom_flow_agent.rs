
use std::error::Error;
use reagent_rs::{
    flow, init_default_tracing, invocations, Agent, AgentBuilder, AgentError, Message 
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    init_default_tracing();
    
    let mut agent = AgentBuilder::default()
        .set_model("qwen3:0.6b")
        .set_system_prompt("You are a helpful, assistant.")
        .set_flow(flow!(custom_flow)) 
        .build()
        .await?;

    let resp = agent.invoke_flow("What is the meaning of life?").await?;
    println!("{resp:#?}");

    Ok(())
}

// you can create own functions as the flows for invoking an agent
// when invoke_flow or invoke_flow_with_template is called,
// this is the function that will override the default flow if the 
// agent
async fn custom_flow(agent: &mut Agent, prompt: String) -> Result<Message, AgentError> {
    agent.history.push(Message::user(prompt));
    let mut last = None;
    for _ in 0..agent.max_iterations.unwrap_or(1) {
        let response = invocations::invoke_without_tools(agent).await?;
        last = Some(response.message);
    }
    Ok(last.unwrap())
}
