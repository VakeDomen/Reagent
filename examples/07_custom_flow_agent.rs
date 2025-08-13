
use std::error::Error;
use reagent::{
    init_default_tracing, 
    Flow, 
    FlowFuture, 
    invocations::invoke_without_tools, 
    Agent, 
    AgentBuilder, 
    Message 
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    init_default_tracing();
    
    let mut agent = AgentBuilder::default()
        .set_model("qwen3:0.6b")
        .set_system_prompt("You are a helpful, assistant.")
        .set_flow(Flow::Custom(custom_flow))
        .build()
        .await?;

    let resp = agent.invoke_flow("What is the meaning of lige?").await?;
    println!("{resp:#?}");

    Ok(())
}

// you can create own functions as the flows for invoking an agent
// when invoke_flow or invoke_flow_with_template is called,
// this is the function that will override the default flow if the 
// agent
// again Box::pin to make the agent clonable. You can return Ok(Message)
// or Err(AgentError) from the fn
fn custom_flow<'a>(agent: &'a mut Agent, prompt: String) -> FlowFuture<'a> {
    Box::pin(async move {
        agent.history.push(Message::user(prompt));
        let mut last_response = None;
        // do your thing
        for _ in 0..agent.max_iterations.unwrap_or(1) {
            let response = invoke_without_tools(agent).await?;
            agent.history.push(response.message.clone());
            last_response = Some(response.message);
        }
        
        Ok(last_response.unwrap())
    })    
}