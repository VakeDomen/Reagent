mod agent;
mod builder;
mod config;
mod error;
mod flows;

pub use agent::*;
pub use builder::*;
pub use config::*;
pub use error::*;
pub use flows::*;

#[tokio::test]
async fn agent_builder_encodes_invoke_input_and_output_types() {
    use crate::{Standard, Structured, TemplateInput};

    #[derive(serde::Deserialize, rmcp::schemars::JsonSchema)]
    struct Entity {
        name: String,
    }

    let _: Agent<Prompt, Standard> = AgentBuilder::default()
        .set_model("test")
        .build()
        .await
        .unwrap();
    let _: Agent<Prompt, Structured<Entity>> = AgentBuilder::default()
        .set_model("test")
        .structured_output::<Entity>()
        .build()
        .await
        .unwrap();
    let _: Agent<TemplateInput, Standard> = AgentBuilder::default()
        .set_model("test")
        .set_template(crate::Template::simple("Hello {{name}}"))
        .build()
        .await
        .unwrap();
}
