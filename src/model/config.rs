use crate::services::llm::InferenceOptions;

/// Reusable model identity and inference defaults.
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

impl From<&ModelConfig> for InferenceOptions {
    fn from(config: &ModelConfig) -> Self {
        Self {
            num_ctx: config.num_ctx,
            repeat_last_n: config.repeat_last_n,
            repeat_penalty: config.repeat_penalty,
            temperature: config.temperature,
            seed: config.seed,
            stop: config.stop.clone(),
            num_predict: config.num_predict,
            max_tokens: None,
            top_k: config.top_k,
            top_p: config.top_p,
            min_p: config.min_p,
            presence_penalty: config.presence_penalty,
            frequency_penalty: config.frequency_penalty,
        }
    }
}

impl ModelConfig {
    pub(crate) fn from_parts(model: String, options: InferenceOptions) -> Self {
        Self {
            model: Some(model),
            temperature: options.temperature,
            top_p: options.top_p,
            presence_penalty: options.presence_penalty,
            frequency_penalty: options.frequency_penalty,
            num_ctx: options.num_ctx,
            repeat_last_n: options.repeat_last_n,
            repeat_penalty: options.repeat_penalty,
            seed: options.seed,
            stop: options.stop,
            num_predict: options.num_predict,
            top_k: options.top_k,
            min_p: options.min_p,
        }
    }
}
