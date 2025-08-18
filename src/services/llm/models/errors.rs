/// Errors that can occur when interacting with the underlying model client.
///
/// These errors typically arise when building requests, communicating with the
/// API, serializing/deserializing payloads, or due to misconfiguration.
#[derive(Debug)]
pub enum ModelClientError {
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

impl std::fmt::Display for ModelClientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelClientError::Request(s) => write!(f, "Request Error: {s}"),
            ModelClientError::Api(s) => write!(f, "API Error: {s}"),
            ModelClientError::Serialization(s) => write!(f, "Serialization Error: {s}"),
            ModelClientError::Config(s) => write!(f, "Config Error: {s}"),
            ModelClientError::Unsupported(s) => write!(f, "Unsupported: {s}"),
        }
    }
}

impl std::error::Error for ModelClientError {}

impl From<reqwest::Error> for ModelClientError {
    fn from(err: reqwest::Error) -> Self { ModelClientError::Request(err.to_string()) }
}
