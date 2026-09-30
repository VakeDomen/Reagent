use std::collections::{BTreeMap, BTreeSet};

use reqwest::{
    header::{HeaderMap, HeaderName, HeaderValue, AUTHORIZATION, CONTENT_TYPE},
    Client,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{services::llm::InferenceClientError, ClientConfig, Provider};

/// Instruction text used by the common System One question methods.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Question(pub String);

impl<T: Into<String>> From<T> for Question {
    fn from(value: T) -> Self {
        Self(value.into())
    }
}

#[derive(Debug, Clone, Default)]
pub struct NoQuestions;

#[derive(Debug, Clone, Default)]
pub struct HasQuestions;

#[derive(Debug, Clone, Default)]
pub(crate) struct QuestionSet {
    pub(crate) questions: BTreeMap<String, SystemOneQuestion>,
    duplicates: BTreeSet<String>,
}

impl QuestionSet {
    pub(crate) fn insert(&mut self, key: String, question: SystemOneQuestion) {
        if self.questions.insert(key.clone(), question).is_some() {
            self.duplicates.insert(key);
        }
    }

    pub(crate) fn validate(&self) -> Result<(), String> {
        if self.questions.is_empty() {
            return Err("at least one question is required".into());
        }
        if let Some(key) = self.duplicates.iter().next() {
            return Err(format!("duplicate question key: {key}"));
        }
        for (key, question) in &self.questions {
            question
                .validate()
                .map_err(|error| format!("{key}: {error}"))?;
        }
        Ok(())
    }
}

/// A question sent to a System One endpoint. `Value` permits the structured
/// instructions and criteria supported by Jev's advanced API.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum SystemOneQuestion {
    Noul {
        instructions: Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        criteria: Option<NoulCriteria>,
    },
    Choice {
        instructions: Value,
        criteria: BTreeMap<String, Value>,
    },
    Score {
        instructions: Value,
        criteria: Vec<Value>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NoulCriteria {
    #[serde(rename = "true")]
    pub when_true: Value,
    #[serde(rename = "false")]
    pub when_false: Value,
}

impl NoulCriteria {
    pub fn new(when_true: impl Into<String>, when_false: impl Into<String>) -> Self {
        Self {
            when_true: Value::String(when_true.into()),
            when_false: Value::String(when_false.into()),
        }
    }
}

impl SystemOneQuestion {
    pub fn noul(instructions: impl Into<Question>) -> Self {
        Self::Noul {
            instructions: Value::String(instructions.into().0),
            criteria: None,
        }
    }

    pub fn noul_with_criteria(instructions: impl Into<Question>, criteria: NoulCriteria) -> Self {
        Self::Noul {
            instructions: Value::String(instructions.into().0),
            criteria: Some(criteria),
        }
    }

    pub fn choice<K, V, C>(instructions: impl Into<Question>, criteria: C) -> Self
    where
        K: Into<String>,
        V: Into<String>,
        C: IntoIterator<Item = (K, V)>,
    {
        Self::Choice {
            instructions: Value::String(instructions.into().0),
            criteria: criteria
                .into_iter()
                .map(|(key, description)| (key.into(), Value::String(description.into())))
                .collect(),
        }
    }

    pub fn score<V, C>(instructions: impl Into<Question>, criteria: C) -> Self
    where
        V: Into<String>,
        C: IntoIterator<Item = V>,
    {
        Self::Score {
            instructions: Value::String(instructions.into().0),
            criteria: criteria
                .into_iter()
                .map(|level| Value::String(level.into()))
                .collect(),
        }
    }

    pub(crate) fn validate(&self) -> Result<(), String> {
        let (instructions, criteria_error) = match self {
            Self::Noul { instructions, .. } => (instructions, None),
            Self::Choice {
                instructions,
                criteria,
            } => (
                instructions,
                (criteria.len() < 2 || criteria.len() > 255)
                    .then_some("choice requires 2 to 255 criteria"),
            ),
            Self::Score {
                instructions,
                criteria,
            } => (
                instructions,
                (criteria.len() < 2 || criteria.len() > 10)
                    .then_some("score requires 2 to 10 criteria"),
            ),
        };
        if let Some(error) = criteria_error {
            return Err(error.into());
        }
        if matches!(instructions, Value::String(text) if text.trim().is_empty()) {
            return Err("instructions cannot be empty".into());
        }
        if !matches!(
            instructions,
            Value::String(_) | Value::Object(_) | Value::Array(_)
        ) {
            return Err("instructions must be text, an object, or an array".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SystemOneRequest {
    pub state: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub questions: BTreeMap<String, SystemOneQuestion>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct SystemOneResponse {
    pub model: String,
    pub answers: BTreeMap<String, SystemOneAnswer>,
    pub usage: SystemOneUsage,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum SystemOneAnswer {
    Noul(NoulAnswer),
    Choice(ChoiceAnswer),
    Score(ScoreAnswer),
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct NoulAnswer {
    pub noul: f64,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct ChoiceAnswer {
    pub choice: String,
    pub confidence: f64,
    pub probabilities: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct ScoreAnswer {
    pub score: f64,
    pub confidence: f64,
    pub legend: BTreeMap<String, Value>,
    pub probabilities: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct SystemOneUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

/// Transport for the native `/v1/systemone` request shape.
#[derive(Debug, Clone)]
pub struct SystemOneClient {
    client: Client,
    base_url: String,
}

impl SystemOneClient {
    pub fn new(config: ClientConfig) -> Result<Self, InferenceClientError> {
        if !matches!(config.provider, None | Some(Provider::SystemOne)) {
            return Err(InferenceClientError::Config(
                "System One requests require the SystemOne provider".into(),
            ));
        }
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        if let Some(api_key) = config.api_key {
            let value = HeaderValue::from_str(&format!("Bearer {api_key}"))
                .map_err(|error| InferenceClientError::Config(error.to_string()))?;
            headers.insert(AUTHORIZATION, value);
        }
        if let Some(extra) = config.extra_headers {
            for (name, value) in extra {
                let name = HeaderName::from_bytes(name.as_bytes())
                    .map_err(|error| InferenceClientError::Config(error.to_string()))?;
                let value = HeaderValue::from_str(&value)
                    .map_err(|error| InferenceClientError::Config(error.to_string()))?;
                headers.insert(name, value);
            }
        }
        let client = Client::builder().default_headers(headers).build()?;
        Ok(Self {
            client,
            base_url: config
                .base_url
                .unwrap_or_else(|| "http://localhost:8080".into()),
        })
    }

    pub async fn evaluate(
        &self,
        request: SystemOneRequest,
    ) -> Result<SystemOneResponse, InferenceClientError> {
        let url = format!("{}/v1/systemone", self.base_url.trim_end_matches('/'));
        let response = self.client.post(url).json(&request).send().await?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await?;
            return Err(InferenceClientError::Api(format!("HTTP {status}: {body}")));
        }
        response
            .json::<SystemOneResponse>()
            .await
            .map_err(|error| InferenceClientError::Serialization(error.to_string()))
    }
}
