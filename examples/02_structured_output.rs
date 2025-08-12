
use std::error::Error;
use reagent::{init_default_tracing, AgentBuilder};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct MyWeatherOuput {
  _windy: bool,
  _temperature: i32,
  _description: String
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    init_default_tracing();

    let mut agent = AgentBuilder::default()
        .set_model("qwen3:0.6b")
        .set_system_prompt("You make up weather info in JSON. You always say it's sowing")
        // If you need structured output you can set 
        // the response format schema of the agent
        .set_response_format(
            r#"
            {
              "type":"object",
              "properties":{
                "windy":{"type":"boolean"},
                "temperature":{"type":"integer"},
                "description":{"type":"string"}
              },
              "required":["windy","temperature","description"]
            }
            "#,
        )
        .build()
        .await?;
    
    // however the "invoke_flow" still returns a general Message 
    // struct and you have to parse the message.content Option<string> to
    // own struct

    // If you call `invoke_flow_structured_output` the agent will return your 
    // deserialized object
    let resp: MyWeatherOuput = agent.invoke_flow_structured_output("What is the current weather in Koper?").await?;
    println!("\n-> Agent: {resp:#?}");

    Ok(())
}
