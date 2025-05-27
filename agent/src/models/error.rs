use ollama_rs::error::OllamaError;


#[derive(Debug)]
pub enum AgentError {
    OllamaError(OllamaError)
}