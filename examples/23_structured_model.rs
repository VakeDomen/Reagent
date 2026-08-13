use reagent_rs::{ChatResponse, JsonSchema, Model};
use serde::Deserialize;
use std::error::Error;

#[derive(Debug, Deserialize, JsonSchema)]
struct Entities {
    people: Vec<String>,
    places: Vec<String>,
}

/// A reusable, stateless extractor. Each `invoke` starts from the model's configured defaults.
#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let extractor = Model::llm("qwen3:0.6b")
        .temperature(0.0)
        .stream(true)
        .structured_output::<Entities>()
        .build()?;

    let response: ChatResponse<Entities> = extractor
        .invoke("Extract entities from: Ada Lovelace visited London.")
        .await?;

    println!("{:#?}", response.message.content);
    Ok(())
}
