use reagent_rs::AgentBuilder;
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    reagent_rs::observability::init_default_tracing();

    // creating agents follows the builder pattern
    let mut agent = AgentBuilder::default()
        // model must be set, everything else has
        // defualts and is optional
        .set_model("qwen3:0.6b")
        .set_system_prompt("You are a helpful assistant.")
        .set_temperature(0.6)
        .set_num_ctx(2048) // lol
        // call build to return the agent
        .build()
        // creation can fail (sever unreachable?)
        .await?;

    // call agents by calling the "invoke_flow" method
    let resp = agent
        .invoke_flow("How do i increase context size in Ollama?")
        .await?;
    println!("Agent: {}", resp.content.unwrap());

    // internally agent holds the conversation histroy
    let resp = agent.invoke_flow("What did you just say?").await?;
    println!("Agent: {}", resp.content.unwrap());

    // but it can be reset
    // system message will stay, other messages will
    // be deleted
    agent.clear_history();

    let resp = agent.invoke_flow("What did you just say?").await?;
    println!("Agent: {resp:#?}");

    Ok(())
}
