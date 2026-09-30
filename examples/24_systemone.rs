use reagent_rs::{Invocation, Model, SystemOneAnswer};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = Model::systemone()
        .noul("refund", "Is the customer requesting a refund?")
        .choice(
            "department",
            "Which department should handle this?",
            [
                ("returns", "Returns and exchanges"),
                ("shipping", "Delivery issues"),
            ],
        )
        .base_url("url")
        .api_key("key")
        .build()?;

    let response = model
        .invoke("I ordered size 10 shoes but received size 8.")
        .await?;
    if let Some(SystemOneAnswer::Noul(answer)) = response.answers.get("refund") {
        println!("Refund probability: {}", answer.noul);
    }

    let response = Invocation::systemone("My package has not arrived.")
        .noul("late", "Is the delivery late?")
        .invoke()
        .await?;
    println!("One-off answers: {:?}", response.answers);
    Ok(())
}
