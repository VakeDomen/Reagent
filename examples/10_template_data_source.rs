
use std::{collections::HashMap, error::Error, future::Future};
use reagent::{init_default_tracing, models::flow::util::templating::{Template, TemplateDataSource}, AgentBuilder};

struct MyCustomDataSource;

impl TemplateDataSource for MyCustomDataSource {
    // will get called to generate the values when invoke_flow_with_template is called
    // used to genrate values on-the-fly
    fn get_values(&self) -> std::pin::Pin<Box<dyn Future<Output = HashMap<String, String>> + Send>> {
        Box::pin(async move {
            HashMap::from([
                ("date".to_string(), "1.1.2025".to_string())
            ])
        })
    }

    // needs to be Clonable
    fn clone_data_source(&self) -> Box<dyn TemplateDataSource> {
        Box::new(MyCustomDataSource)
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    init_default_tracing();
    
    let data_source = MyCustomDataSource;
    let template = Template::new(r#"
            Today is: {{date}}
            Answer the following question: {{question}}
        "#,
        data_source
    );

    let mut agent = AgentBuilder::default()
        .set_model("qwen3:0.6b")
        .set_template(template)
        .build()
        .await?;

    let prompt_data = HashMap::from([
        ("question", "What's the date today?")
    ]);

    let resp = agent.invoke_flow_with_template(prompt_data).await?;
    println!("\t-> Agent: {:#?}", resp);

    Ok(())
}

