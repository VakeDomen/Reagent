use reagent_rs::{InvocationBuilder, Message};
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    reagent_rs::observability::init_default_tracing();

    let req = InvocationBuilder::default()
        .model("ministral-3:14b")
        .set_message(Message::system("You are short and concise"))
        .stream(true);

    let _ = req
        .clone()
        .add_message(Message::user("What's the meaning of life?"))
        .invoke()
        .await;

    let _ = req
        .add_message(Message::user("Who is the owner of Nvidia?"))
        .invoke()
        .await;

    Ok(())
}
