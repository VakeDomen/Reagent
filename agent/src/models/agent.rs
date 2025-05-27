use ollama_rs::{generation::chat::{request::ChatMessageRequest, ChatMessage, ChatMessageResponse}, Ollama};
use anyhow::Result;

use super::AgentError;


#[derive(Debug)]
pub struct Agent {
    model: String,
    history: Vec<ChatMessage>,
    ollama_client: Ollama,
}

impl Agent {
    pub(crate) fn new(
        model: &str,
        ollama_host: &str,
        ollama_port: u16,
        system_prompt: &str,
    ) -> Self {
        let mut history = vec![];
        history.push(ChatMessage::system(system_prompt.into()));


        Self { 
            model: model.into(), 
            history,
            ollama_client: Ollama::new(ollama_host, ollama_port)
        }
    }


    pub async fn invoke<T>(&mut self, prompt: T) -> Result<ChatMessageResponse, AgentError> where T: Into<String> {
        let resp = self.ollama_client.send_chat_messages_with_history(
            &mut self.history, 
            ChatMessageRequest::new(
                self.model.clone(),
                vec![ChatMessage::user(prompt.into())], 
            ),
        ).await;
        
        match resp {
            Ok(r) => Ok(r),
            Err(e) => Err(AgentError::OllamaError(e)),
        }
    }
}