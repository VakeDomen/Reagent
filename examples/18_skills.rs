use reagent_rs::{AgentBuilder, NotificationContent};
use std::error::Error;

static PROMPT: &str = r"
Summarize below notes please:

An agent harness wraps a language model so it can do more than just complete text.
It usually manages the system prompt, conversation history, tool calling, structured
output, retries, tracing, and model configuration. The harness can also expose specialized
skills, such as summarization or code review, without loading all instructions into the
context at once.

The goal is to make model behavior more predictable and easier to integrate into real
applications. Instead of scattering prompt logic everywhere, the harness centralizes how
the agent is built, invoked, and observed. This is useful when working with different
providers, local models, MCP tools, or custom flows.";

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let (mut agent, mut notification_reciever) = AgentBuilder::default()
        .set_model("qwen3:4b-instruct")
        .add_skill_collection("./examples/skills")
        .build_with_notification()
        .await?;

    let handle = tokio::spawn(async move {
        while let Some(msg) = notification_reciever.recv().await {
            match msg.content {
                NotificationContent::ToolCallRequest(t) => {
                    println!("Called tool: {}", t.function.name)
                }
                NotificationContent::ToolCallErrorResult(t) => {
                    println!("Error calling tool: {}", t)
                }
                _ => (),
            };
        }
    });

    let resp = agent.invoke_flow(PROMPT).await?;

    println!("{}", resp.content.unwrap());

    drop(agent);
    handle.await?;
    Ok(())
}
