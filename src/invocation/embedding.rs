use super::Invocation;

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
}
