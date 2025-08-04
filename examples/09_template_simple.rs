
use std::{collections::HashMap, error::Error};
use reagent::{
    init_default_tracing, util::Template, AgentBuilder 
};
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
    

    // you can define your prompt template
    // the {{values}} will be replaced by your 
    // values when prompted. 
    let template = Template::simple(r#"
    Yout name is {{name}}
    Answer the following question: {{question}}
    "#);

    // build the agent
    let mut agent = AgentBuilder::default()
        .set_model("qwen3:0.6b")
        .set_template(template)
        .build()
        .await?;

    // we won't pass the string as a prompt but HashMap of values
    // the map is <K, V> where K and V are Into<String> so you can
    // pass your own types
    let prompt_data = HashMap::from([
        ("name", "Gregor"),
        ("question", "What's your name?")
    ]);

    // if you are using a template invoke the agent using
    // invoke_flow_with_template instead invoke_flow
    // you pass it the HashMap of values
    let resp = agent.invoke_flow_with_template(prompt_data).await?;
    println!("\n-> Agent: {}", resp.content.unwrap_or_default());



    // ...custom structs that implement Into<String>
    let name = MyCustomDataHolder { 
        value: "Peter".into()
    };
    let question = MyCustomDataHolder { 
        value: "What's your name?".into() 
    };

    let prompt_data = HashMap::from([
        ("name", name),
        ("question", question)
    ]);

    let resp = agent.invoke_flow_with_template(prompt_data).await?;
    println!("\n-> Agent: {}", resp.content.unwrap_or_default());

    Ok(())
}

