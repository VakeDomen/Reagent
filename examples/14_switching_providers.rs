
use std::error::Error;
use reagent::{init_default_tracing, json, AgentBuilder, Provider};
use schemars::{schema_for, JsonSchema};
use serde::Deserialize;


#[derive(Debug, Deserialize, JsonSchema)]
struct MyWeatherOuput {
  _windy: bool,
  _temperature: i32,
  _description: String
}


#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    init_default_tracing();
    
    let agent = AgentBuilder::default()
        .set_model("qwen3:0.6b")
        // ollama is set as default you don't 
        // actually have to set it
        .set_provider(Provider::Ollama)
        .set_base_url("http://myollamapi.example.com:11456")
        .set_api_key("MY_API_KEY")
        .build()
        .await?;

    let agent = AgentBuilder::default()
        .set_model("mistralai/mistral-small-3.2-24b-instruct:free")
        // Currently open router is the only one
        // supported outside of ollama
        // you can specify other but the agent will 
        // fail to build, throwing "unsupported" error
        .set_provider(Provider::OpenRouter)
        .set_api_key("MY_API_KEY")
        .build()
        .await?;


    // careful, different providers may need diffrent settings  
    // like response format
    let schema = schema_for!(MyWeatherOuput);
    let open_router_response_format = serde_json::to_string_pretty(&json!({
      "type": "json_schema",
      "json_schema": {
        "name": "MyWeatherOuput",
        "schema": schema
      }
    }))?;

    let mut agent = AgentBuilder::default()
        // .set_model("qwen3:0.6b")
        .set_system_prompt("You make up weather info in JSON. You always say it's sowing")
        // you can also use the schemars with serde to construct schema from struct
        .set_model("mistralai/mistral-small-3.2-24b-instruct:free")
        .set_provider(reagent::Provider::OpenRouter)
        .set_api_key("MY_API_KEY")
        .set_response_format(open_router_response_format)
        .set_stream(true)
        .build()
        .await?;

    let resp: MyWeatherOuput = agent.invoke_flow_structured_output("What is the current weather in Koper?").await?;
    println!("\n-> Agent: {resp:#?}");


    Ok(())
}
