use reagent_rs::{AgentBuilder, NotificationContent};
use std::error::Error;
use std::io::{self, Write};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let (mut agent, mut notification_reciever) = AgentBuilder::default()
        .set_model("hf.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:UD-Q4_K_XL")
        .set_system_prompt(
            "You are an autonomous agent that performs actions. Use
            tools and skills until you are able to answer user's questions or complete the tasks.",
        )
        .add_skill_collection("./examples/skills")
        .add_bash()?
        .set_stream(true)
        .build_with_notification()
        .await?;

    let handle = tokio::spawn(async move {
        while let Some(msg) = notification_reciever.recv().await {
            match msg.content {
                NotificationContent::ToolCallRequest(t) => {
                    println!(
                        "\t-> Called tool: {} ({})",
                        t.function.name, t.function.arguments
                    )
                }
                NotificationContent::ToolCallErrorResult(t) => {
                    println!("Error calling tool: {}", t)
                }
                NotificationContent::Token(t) => {
                    print!("{}", t.value);
                    io::stdout().flush().unwrap();
                }
                _ => (),
            };
        }
    });

    println!("What do you want me to do?");
    println!("Write /bye to exit");
    loop {
        println!("\n\nUser: ");
        let prompt = user_input();
        if prompt.eq("/bye") {
            println!("Bye");
            break;
        }

        println!("\n\n Agent: ");
        let _ = agent.invoke_flow(prompt).await?;
    }

    // println!("{}", resp.content.unwrap());

    drop(agent);
    handle.await?;
    Ok(())
}

fn user_input() -> String {
    let mut line = String::new();
    std::io::stdin().read_line(&mut line).unwrap();
    line.trim().to_string()
}
