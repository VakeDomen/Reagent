use crate::{
    services::llm::{models::embedding::EmbeddingsRequest, ClientBuilder},
    EmbeddingsResponse, Invocation, InvocationError,
};

/// Provider-neutral input for one embedding call.
#[derive(Debug, Clone, Default)]
pub struct Embedding {
    pub(crate) input: Vec<String>,
    pub(crate) keep_alive: Option<String>,
}

pub type EmbeddingInvocation = Invocation<Embedding>;

impl Invocation<Embedding> {
    pub fn embedding(input: impl Into<String>) -> Self {
        Self::new(Embedding {
            input: vec![input.into()],
            keep_alive: None,
        })
    }

    pub fn embeddings<T, I>(inputs: I) -> Self
    where
        T: Into<String>,
        I: IntoIterator<Item = T>,
    {
        Self::new(Embedding {
            input: inputs.into_iter().map(Into::into).collect(),
            keep_alive: None,
        })
    }

    pub fn keep_alive(mut self, value: impl Into<String>) -> Self {
        self.request.keep_alive = Some(value.into());
        self
    }

    pub async fn invoke(self) -> Result<EmbeddingsResponse, InvocationError> {
        if self.request.input.is_empty() {
            return Err(InvocationError::InputNotDefined);
        }
        let model = self.model.ok_or(InvocationError::ModelNotDefined)?;
        let client = self.client_config.build()?;
        Ok(client
            .embeddings(EmbeddingsRequest {
                model,
                input: self.request.input,
                options: None,
                keep_alive: self.request.keep_alive,
            })
            .await?)
    }
}
