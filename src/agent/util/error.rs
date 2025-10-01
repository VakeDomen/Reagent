use crate::services::llm::InferenceClientError;

#[derive(Debug)]
pub enum InvocationError {
    ModelNotDefined,
    InferenceError(InferenceClientError),
}

impl From<InferenceClientError> for InvocationError {
    fn from(err: InferenceClientError) -> Self {
        InvocationError::InferenceError(err)
    }
}

impl std::error::Error for InvocationError {}

impl std::fmt::Display for InvocationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InvocationError::ModelNotDefined => write!(f, "Inference model not defined"),
            InvocationError::InferenceError(inference_client_error) => {
                write!(f, "Client error during inference: {inference_client_error}")
            }
        }
    }
}
