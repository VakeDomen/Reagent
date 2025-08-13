use crate::{templates::Template, McpServerType, Tool};


#[derive(Debug, Clone, Default)]
pub struct OllamaConfig {
    pub ollama_url: Option<String>
}


#[derive(Debug, Clone, Default)]
pub struct ModelConfig {
    pub model: Option<String>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub num_ctx: Option<u32>,
    pub repeat_last_n: Option<i32>,
    pub repeat_penalty: Option<f32>,
    pub seed: Option<i32>,
    pub stop: Option<String>,
    pub num_predict: Option<i32>,
    pub top_k: Option<u32>,
    pub min_p: Option<f32>,
}

#[derive(Debug, Clone, Default)]
pub struct PromptConfig {
    pub template: Option<Template>,
    pub system_prompt: Option<String>,
    pub tools: Option<Vec<Tool>>,
    pub response_format: Option<String>,
    pub mcp_servers: Option<Vec<McpServerType>>,
    pub stop_prompt: Option<String>,
    pub stopword: Option<String>,
    pub strip_thinking: Option<bool>,
    pub max_iterations: Option<usize>,
    pub clear_histroy_on_invoke: Option<bool>,
    pub stream: bool,
}