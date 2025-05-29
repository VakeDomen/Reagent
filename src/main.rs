
use std::time::SystemTime;

use agent::{models::AgentBuilder, ToolBuilder};
use anyhow::Result;
use ollama_rs::generation::tools::Tool;

#[tokio::main]
async fn main() -> Result<()> {

    let get_weather_tool = ToolBuilder::new()
        .function_name("get_current_weather")
        .function_description("Get the current weather for a specific location")
        .add_property("location", "string", "The city and state, e.g., San Francisco, CA")
        .add_required_property("location")
        .add_property("unit", "string", "Temperature unit (celsius or fahrenheit)") // Optional parameter
        .build()?; // The '?' will propagate ToolBuilderError if it occurs.
 

    let mut agent = AgentBuilder::default()
        .set_model("mistral-nemo")
        .set_system_prompt("You are an agent that responds with 'Woah'")
        .set_ollama_endpoint("http://hivecore.famnit.upr.si")
        .set_ollama_port(6666)
        .add_tool(get_weather_tool)
        .build();

    println!("Agent: {:#?}", agent);
        
    let resp = agent.invoke("Can you say 'Yeah'").await;
    println!("Agent Resp: {:#?}", resp);

    let resp = agent.invoke("Did you say 'woah' or 'yeah'?").await;
    println!("Agent Resp: {:#?}", resp);


    let resp = agent.invoke("What is the current weather in Ljubljana?").await;
    println!("Agent Resp: {:#?}", resp);


    println!("Agent: {:#?}", agent);

    Ok(())
}
