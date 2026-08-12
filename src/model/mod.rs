mod builder;
mod execution;
mod model;

pub use builder::*;
pub use model::{Embedding, EmbeddingModel, Llm, LlmModel, Model};

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
