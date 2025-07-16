
use std::error::Error;
use reagent::{init_default_tracing, AgentBuilder};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    init_default_tracing();

    let mut agent = AgentBuilder::default()
        .set_model("qwen3:0.6b")
        .set_system_prompt("You make up weather info in JSON. You always say it's sowing")
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
    

    let resp = agent.invoke_flow("What is the current weather in Koper?").await?;
    println!("\n-> Agent: {}", resp.content.unwrap_or_default());

    Ok(())
}
