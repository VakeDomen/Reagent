mod builder;
pub(crate) mod execution;
mod model;

pub use builder::*;
pub use model::{Embedding, EmbeddingModel, IntoModelInput, Llm, LlmModel, Model};

#[test]
fn model_requires_an_identifier() {
    use crate::InvocationError;

    assert!(matches!(
        ModelBuilder::<Llm>::default().build(),
        Err(InvocationError::ModelNotDefined)
    ));
}

#[test]
fn model_keeps_reusable_defaults_but_no_call_input() {
    let model: Model<Llm> = Model::llm("test-model")
        .temperature(0.2)
        .stream(true)
        .build()
        .unwrap();
    assert_eq!(model.id(), "test-model");
    assert_eq!(model.options().temperature, Some(0.2));
    assert!(model.stream_by_default());
}

#[test]
fn model_kind_is_selected_at_build_time() {
    let _: Model<Llm> = Model::llm("chat-model").build().unwrap();
    let _: Model<Embedding> = Model::embedding("embedding-model").build().unwrap();
}

#[test]
fn configured_history_is_copied_into_each_call_without_mutating_the_model() {
    let mut model = Model::llm("chat-model").build().unwrap();
    model
        .set_history(vec![crate::Message::system("be concise")])
        .set_temperature(0.2);

    assert_eq!(model.history().len(), 1);
    assert_eq!(model.options().temperature, Some(0.2));
}

#[test]
fn structured_output_is_encoded_in_the_model_type() {
    use crate::Structured;

    #[derive(serde::Deserialize, rmcp::schemars::JsonSchema)]
    struct Entity {
        name: String,
    }

    let _: Model<Llm, Structured<Entity>> = Model::llm("extractor")
        .structured_output::<Entity>()
        .build()
        .unwrap();
}
