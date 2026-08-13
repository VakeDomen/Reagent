use reagent_rs::{ChatResponse, Invocation, JsonSchema, Message};
use serde::Deserialize;
use std::error::Error;

#[derive(Debug, Deserialize, JsonSchema)]
struct Entities {
    people: Vec<String>,
    places: Vec<String>,
}

/// A one-off request whose response is parsed by the invocation itself.
#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let response: ChatResponse<Entities> = Invocation::chat()
        .model("qwen3:0.6b")
        .message(Message::system(
            "Extract named entities. Return only data matching the supplied JSON schema.",
        ))
        .message(Message::user("Ada Lovelace visited London."))
        .stream(true)
        .response_format_from::<Entities>()
        .invoke()
        .await?;

    println!("{:#?}", response.message.content);
    Ok(())
}
