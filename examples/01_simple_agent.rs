
use std::error::Error;
use reagent::{init_default_tracing, AgentBuilder};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    init_default_tracing();
    

    let mut agent = AgentBuilder::default()
        .set_model("qwen3:0.6b")
        .set_system_prompt("You are a helpful assistant.")
        .set_temperature(0.6)
        .set_num_ctx(2048) // lol
        .build()
        .await?;

    let resp = agent.invoke_flow("How do i increase context size in Ollama?").await?;
    println!("\n-> Agent: {}", resp.content.unwrap_or_default());


    let resp = agent.invoke_flow("What did you just say?").await?;
    println!("\n-> Agent: {}", resp.content.unwrap_or_default());

    agent.clear_history();

    let resp = agent.invoke_flow("What did you just say?").await?;
    println!("\n-> Agent: {}", resp.content.unwrap_or_default());


    Ok(())
}
