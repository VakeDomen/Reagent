
use agent::models::{AgentBuilder};
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {

    let mut agent = AgentBuilder::default()
        .set_model("granite3-moe")
        .set_system_prompt("You are an agent that responds with 'Woah'")
        .build();

    println!("Agent: {:#?}", agent);
        
    let resp = agent.invoke("Hello").await;

    println!("Agent Resp: {:#?}", resp);
    Ok(())
}
