use reagent_rs::Model;
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let model = Model::embedding("bge-m3").build()?;

    let resp = model.invoke("Reagent builds AI agents in Rust.").await?;

    println!("embedding count: {}", resp.embeddings.len());
    println!("first embedding dimensions: {}", resp.embedding.len());

    let resp = model
        .invoke([
            "Reagent builds AI agents in Rust.",
            "Embeddings turn text into vectors.",
        ])
        .await?;

    println!("embedding count: {}", resp.embeddings.len());
    println!("first embedding dimensions: {}", resp.embedding.len());

    Ok(())
}
