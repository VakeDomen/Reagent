use reagent_rs::{Invocation, Message, Model};
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    reagent_rs::observability::init_default_tracing();

    let model = Model::llm("qwen3:0.6b").stream(true).build()?;
    let _ = model.invoke("Who is the owner of Nvidia?").await;
    let _ = model.invoke("What's the meaning of life?").await;

    let request = Invocation::chat()
        .model("qwen3:0.6b")
        .message(Message::system("You are short and concise"))
        .message(Message::user("Explain stateless models."));
    let _ = request.invoke().await;

    Ok(())
}
