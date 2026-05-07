use serde::{Deserialize, Serialize, Serializer};
use serde_json::Value;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    Developer,
    User,
    Assistant,
    Tool,
}

#[derive(Serialize, Debug, Clone, Default, Deserialize)]
pub struct BaseRequest {
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<Value>,

    #[serde(flatten)]
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "options")]
    #[serde(serialize_with = "serialize_options_as_map")]
    pub options: Option<InferenceOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct InferenceOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_ctx: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat_last_n: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_predict: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
}

impl InferenceOptions {
    pub fn is_empty(&self) -> bool {
        self.num_ctx.is_none()
            && self.repeat_last_n.is_none()
            && self.repeat_penalty.is_none()
            && self.temperature.is_none()
            && self.seed.is_none()
            && self.stop.is_none()
            && self.num_predict.is_none()
            && self.max_tokens.is_none()
            && self.top_k.is_none()
            && self.top_p.is_none()
            && self.min_p.is_none()
            && self.presence_penalty.is_none()
            && self.frequency_penalty.is_none()
    }

    pub fn into_option(self) -> Option<Self> {
        if self.is_empty() {
            None
        } else {
            Some(self)
        }
    }

    pub fn merge_over(self, defaults: Self) -> Self {
        Self {
            num_ctx: self.num_ctx.or(defaults.num_ctx),
            repeat_last_n: self.repeat_last_n.or(defaults.repeat_last_n),
            repeat_penalty: self.repeat_penalty.or(defaults.repeat_penalty),
            temperature: self.temperature.or(defaults.temperature),
            seed: self.seed.or(defaults.seed),
            stop: self.stop.or(defaults.stop),
            num_predict: self.num_predict.or(defaults.num_predict),
            max_tokens: self.max_tokens.or(defaults.max_tokens),
            top_k: self.top_k.or(defaults.top_k),
            top_p: self.top_p.or(defaults.top_p),
            min_p: self.min_p.or(defaults.min_p),
            presence_penalty: self.presence_penalty.or(defaults.presence_penalty),
            frequency_penalty: self.frequency_penalty.or(defaults.frequency_penalty),
        }
    }
}

fn serialize_options_as_map<S>(
    options: &Option<InferenceOptions>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    match options {
        Some(opts) => {
            let map = serde_json::to_value(opts)
                .map_err(serde::ser::Error::custom)?
                .as_object()
                .cloned()
                .unwrap_or_default();
            map.serialize(serializer)
        }
        None => serializer.serialize_none(),
    }
}
