
use std::error::Error;
use reagent_rs::{
    flow, invoke_without_tools, Agent, AgentBuilder, AgentError, Message 
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    
    let mut agent = AgentBuilder::default()
        .set_model("qwen3:0.6b")
        .set_flow(flow!(custom_flow)) 
        .build()
        .await?;

    // there is a HashMap<String, serde_json::Value> to store any data
    // for custom states and such
    let tokens = agent.state.get("last_token_count");
    println!("Number of tokens in last answer: {:#?}", tokens);

    let _ = agent.invoke_flow("What is the meaning of life?").await?;

    // there is a HashMap<String, serde_json::Value> to store any data
    // for custom states and such
    let tokens = agent.state.get("last_token_count");
    println!("Number of tokens in last answer: {:#?}", tokens);
    Ok(())
}

async fn custom_flow(agent: &mut Agent, prompt: String) -> Result<Message, AgentError> {
    agent.history.push(Message::user(prompt));
    let response = invoke_without_tools(agent).await?;
    
    
    // insert into agent state
    if let Some(tokens) = response.eval_count {
      agent.state.insert(
        "last_token_count".into(), 
        tokens.into()
      );
    }

    Ok(response.message)
}
