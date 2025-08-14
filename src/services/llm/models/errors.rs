#[derive(Debug)]
pub enum ModelClientError {
    Request(String),
    Api(String),
    Serialization(String),
    Config(String),
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
