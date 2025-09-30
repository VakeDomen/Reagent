/// Errors that can occur when interacting with the underlying model client.
///
/// These errors typically arise when building requests, communicating with the
/// API, serializing/deserializing payloads, or due to misconfiguration.
#[derive(Debug)]
pub enum InferenceClientError {
    /// Failure constructing or sending a request (e.g. network issue).
    Request(String),
    /// Error returned directly from the model provider’s API.
    Api(String),
    /// Failure serializing a request or deserializing a response.
    Serialization(String),
    /// Invalid or missing client configuration.
    Config(String),
    /// Attempted to use a feature not supported by the provider or client.
    Unsupported(String),
}

impl std::fmt::Display for InferenceClientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InferenceClientError::Request(s) => write!(f, "Request Error: {s}"),
            InferenceClientError::Api(s) => write!(f, "API Error: {s}"),
            InferenceClientError::Serialization(s) => write!(f, "Serialization Error: {s}"),
            InferenceClientError::Config(s) => write!(f, "Config Error: {s}"),
            InferenceClientError::Unsupported(s) => write!(f, "Unsupported: {s}"),
        }
    }
}

impl std::error::Error for InferenceClientError {}

impl From<reqwest::Error> for InferenceClientError {
    fn from(err: reqwest::Error) -> Self {
        InferenceClientError::Request(err.to_string())
    }
}
