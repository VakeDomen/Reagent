
#[derive(Debug)]
pub enum OllamaError {
    Request(String),
    Api(String),
    Serialization(String),
}

impl std::fmt::Display for OllamaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OllamaError::Request(s) => write!(f, "Request Error: {s}"),
            OllamaError::Api(s) => write!(f, "API Error: {s}"),
            OllamaError::Serialization(s) => write!(f, "Serialization Error: {s}"),
        }
    }
}


impl std::error::Error for OllamaError {}
