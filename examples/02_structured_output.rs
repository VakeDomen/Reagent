use reagent_rs::{AgentBuilder, JsonSchema};
use serde::{Deserialize, Serialize};
use std::error::Error;

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct MyWeatherOuput {
    windy: bool,
    temperature: i32,
    description: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    reagent_rs::observability::init_default_tracing();

    let mut agent = AgentBuilder::default()
        .set_model("qwen3:0.6b")
        .set_system_prompt("You make up weather info in JSON. You always say it's sowing")
        .structured_output::<MyWeatherOuput>()
        .build()
        .await?;

    let resp = agent
        .invoke("What is the current weather in Koper?")
        .await?;
    println!("Agent: {:#?}", resp.content);

    let mut agent = AgentBuilder::default()
        .set_model("qwen3:0.6b")
        .set_system_prompt("You make up weather info in JSON. You always say it's sowing")
        // you can also use the schemars with serde to construct schema from struct
        .structured_output::<MyWeatherOuput>()
        .build()
        .await?;

    let resp = agent
        .invoke("What is the current weather in Koper?")
        .await?;
    println!("Agent: {:#?}", resp.content);

    Ok(())
}
