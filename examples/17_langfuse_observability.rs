use reagent_rs::{
    observability::langfuse::LangfuseOptions, AgentBuilder, InvocationBuilder, Message,
};
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
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
        .temperature(1.0)
        .top_k(30)
        .top_p(0.8)
        .invoke()
        .await;

    let mut agent = AgentBuilder::default()
        .set_model("qwen3:0.6b")
        .set_system_prompt("You are a helpful assistant.")
        .set_temperature(0.6)
        .set_num_ctx(2048)
        .set_stream(true)
        .build()
        .await?;

    let resp = agent
        .invoke_flow(resp.unwrap().message.content.unwrap())
        .await?;
    println!("Agent: {}", resp.content.unwrap());

    // shutdown will flush any unsent buffered traces
    println!("{:#?}", provider.shutdown());
    Ok(())
}
