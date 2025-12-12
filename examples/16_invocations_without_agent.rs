use reagent_rs::{init_default_tracing, InvocationBuilder, Message};
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    init_default_tracing();

    let req = InvocationBuilder::default()
        .model("ministral-3:14b")
        .set_message(Message::system("You are short and concise"))
        .stream(true);

    let resp = req
        .clone()
        .add_message(Message::user("What's the meaning of life?"))
        .invoke()
        .await;
    println!("{:#?}", resp);

    let resp = req
        .add_message(Message::user("Who is the owner of Nvidia?"))
        .invoke()
        .await;
    println!("{:#?}", resp);

    Ok(())
}
