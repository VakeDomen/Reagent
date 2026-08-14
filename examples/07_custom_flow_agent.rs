use reagent_rs::{flow, Agent, AgentBuilder, AgentError, Message};
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    reagent_rs::observability::init_default_tracing();

    let mut agent = AgentBuilder::default()
        .set_model("qwen3:0.6b")
        .set_system_prompt("You are a helpful, assistant.")
        .set_flow(flow!(custom_flow))
        .build()
        .await?;

    let resp = agent.invoke("What is the meaning of life?").await?;
    println!("{resp:#?}");

    Ok(())
}

// you can create own functions as the flows for invoking an agent
// when invoke_flow or invoke_flow_with_template is called,
// this is the function that will override the default flow if the
// agent
async fn custom_flow(agent: &mut Agent, prompt: String) -> Result<Message, AgentError> {
    let mut last = None;
    for iteration in 0..agent.max_iterations.unwrap_or(1) {
        if iteration == 0 {
            agent.history.push(Message::user(prompt.clone()));
        }
        agent.model.set_history(agent.history.clone());
        let response = agent.model.invoke(()).await?;
        agent.history.push(response.message.clone());
        last = Some(response.message);
    }

    Ok(last.unwrap())
}
