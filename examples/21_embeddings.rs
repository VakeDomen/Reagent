use reagent_rs::InvocationBuilder;
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let resp = InvocationBuilder::embedding()
        .model("bge-m3")
        .input("Reagent builds AI agents in Rust.")
        .invoke()
        .await?;

    println!("embedding count: {}", resp.embeddings.len());
    println!("first embedding dimensions: {}", resp.embedding.len());

    let resp = InvocationBuilder::embedding()
        .model("bge-m3")
        .inputs([
            "Reagent builds AI agents in Rust.",
            "Embeddings turn text into vectors.",
        ])
        .invoke()
        .await?;

    println!("embedding count: {}", resp.embeddings.len());
    println!("first embedding dimensions: {}", resp.embedding.len());

    Ok(())
}
