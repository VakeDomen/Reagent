
use std::{collections::HashMap, error::Error};
use reagent::{init_default_tracing, util::templating::Template, AgentBuilder};

struct MyCustomDataHolder {
    pub value: String,
}

impl Into<String> for MyCustomDataHolder {
    fn into(self) -> String {
        self.value
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    init_default_tracing();
    

    let template = Template::simple(r#"
    Yout name is {{name}}
    Answer the following question: {{question}}
    "#);

    let mut agent = AgentBuilder::default()
        .set_model("qwen3:0.6b")
        .set_template(template)
        .build()
        .await?;

    let mut prompt_data = HashMap::new();
    prompt_data.insert("name", "Gregor");
    prompt_data.insert("question", "How do you do?");

    let _ = agent.invoke_flow_with_template(prompt_data).await?;


    let mut prompt_data = HashMap::new();
    prompt_data.insert("name", MyCustomDataHolder { 
        value: String::from("Peter") 
    });
    prompt_data.insert("question", MyCustomDataHolder { 
        value: String::from("Can I fill template from custom structs?") 
    });

    let _ = agent.invoke_flow_with_template(prompt_data).await?;


    println!("{:#?}", agent.history);

    Ok(())
}

