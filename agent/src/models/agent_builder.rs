use crate::services::ollama::models::Tool;

use super::Agent;

#[derive(Debug, Default)]
pub struct AgentBuilder {
    model: Option<String>,
    ollama_url: Option<String>,
    ollama_port: Option<u16>,
    system_prompt: Option<String>,
    tools: Option<Vec<Tool>>
}


impl AgentBuilder { 
    pub fn set_model<T>(mut self, model: T) -> Self where T: Into<String> {
        self.model = Some(model.into());
        self
    }

    pub fn set_ollama_endpoint<T>(mut self, url: T) -> Self where T: Into<String> {
        self.ollama_url = Some(url.into());
        self
    }

    pub fn set_ollama_port(mut self, port: u16) -> Self {
        self.ollama_port = Some(port);
        self
    }

    pub fn set_system_prompt<T>(mut self, prompt: T) -> Self where T: Into<String> {
        self.system_prompt = Some(prompt.into());
        self
    }
    
    pub fn add_tool(mut self, tool: Tool) -> Self {
        match self.tools.as_mut() {
            Some(vec_tools) => vec_tools.push(tool),
            None => self.tools = Some(vec![tool]),
        };
        self
    }

    pub fn build(self) -> Agent {
        let model = match self.model {
            Some(m) => m,
            None => "qwen3:30b".into(),
        };
        
        let ollama_url = match self.ollama_url {
            Some(m) => m,
            None => "http://localhost".into(),
        };
        
        let ollama_port = match self.ollama_port {
            Some(m) => m,
            None => 11434,
        };
        
        let system_prompt = match self.system_prompt {
            Some(m) => m,
            None => "You are a helpful agent.".into(),
        };
        

        Agent::new(&model, &ollama_url, ollama_port, &system_prompt, self.tools)
    }
}