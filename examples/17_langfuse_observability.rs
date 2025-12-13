use reagent_rs::{observability::langfuse::LangfuseOptions, InvocationBuilder, Message};
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // returns the provider so it may be flushed on
    // short running tasks
    let provider = reagent_rs::observability::langfuse::init(LangfuseOptions {
        // made up credentials
        public_key: Some("pk-lf-48df9377-5e49-47a3-bc33-ffd2fbbe16d2"),
        secret_key: Some("sk-lf-03b7df70-262c-4efe-bf1c-208b5be62d39"),
        host: Some("http://localhost:3000"),
    });

    let req = InvocationBuilder::default()
        .model("ministral-3:14b")
        .set_message(Message::system("You are short and creative"))
        .stream(true);

    let resp = req
        .clone()
        .add_message(Message::user("Ask me a question"))
        .temperature(0.8)
        .invoke()
        .await;

    let _ = req
        .add_message(Message::user(resp.unwrap().message.content.unwrap()))
        .invoke()
        .await;

    // shutdown will flush any unsent buffered traces
    println!("{:#?}", provider.shutdown());
    Ok(())
}
