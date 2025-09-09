
use std::{collections::HashMap, error::Error, future::Future};
use reagent_rs::{init_default_tracing, templates::{Template, TemplateDataSource}, AgentBuilder};


// sometimes you want to template values that should be generated on 
// invocation, but don't want to pass it as a parameter to the 
// invoke_flow_with_template every time. You can define a custom
// TemplateDataSource that will generate for the template at invocation
// can will be called in the background and you don't have to pass it every
// time. Usually this is things like current date...
struct MyCustomDataSource;

impl TemplateDataSource for MyCustomDataSource {
    // will get called to generate the values when invoke_flow_with_template is called
    // used to genrate values on-the-fly
    fn get_values(&self) -> std::pin::Pin<Box<dyn Future<Output = HashMap<String, String>> + Send>> {
        Box::pin(async move {
            // shoud return HashMap<String, String>
            HashMap::from([
                ("date".to_string(), "1.1.2025".to_string())
            ])
        })
    }

    // needs to be Clonable so the agent can be Clonable
    fn clone_data_source(&self) -> Box<dyn TemplateDataSource> {
        Box::new(MyCustomDataSource)
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    init_default_tracing();
    
    let data_source = MyCustomDataSource;

    // pass the template and the TemplateDataSource 
    let template = Template::new(r#"
            Today is: {{date}}
            Answer the following question: {{question}}
        "#,
        data_source
    );

    // build the agent
    let mut agent = AgentBuilder::default()
        .set_model("qwen3:0.6b")
        .set_template(template)
        .build()
        .await?;

    // constuct params
    let prompt_data = HashMap::from([
        ("question", "What's the date today?")
    ]);

    // invoke
    let resp = agent.invoke_flow_with_template(prompt_data).await?;
    println!("\t-> Agent: {resp:#?}");

    Ok(())
}

