use serde::{Deserialize, Serialize};

use super::base::BaseRequest;

/// Request for the `/api/generate` endpoint.
#[derive(Serialize, Debug, Clone)]
pub struct GenerateRequest {
    #[serde(flatten)]
    pub base: BaseRequest,
    pub prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub template: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<Vec<i64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw: Option<bool>,
}

/// Response from the `/api/generate` endpoint.
///
/// This structure represents a single response object. If streaming is disabled,
/// it contains the full response. If streaming is enabled, multiple `GenerateResponse`
/// objects will be received, with the final one containing the performance statistics.
#[derive(Deserialize, Debug, Clone)]
pub struct GenerateResponse {
    /// The model name used for generation.
    pub model: String,
    /// The timestamp when the response was created.
    pub created_at: String,
    /// The generated response content. This will be an aggregation if `stream` is false.
    pub response: String,
    /// Indicates if this is the final response (`true`) or part of a stream (`false`).
    pub done: bool,
    /// A reason for why the generation finished. This is only present when `done` is `true`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub done_reason: Option<String>,
    /// An encoding of the conversation context. This can be sent in the next request
    /// to maintain conversational memory. Present only if `done` is `true`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<Vec<i64>>,
    /// Time spent generating the response (nanoseconds). Present only if `done` is `true`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_duration: Option<u64>,
    /// Time spent loading the model (nanoseconds). Present only if `done` is `true`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub load_duration: Option<u64>,
    /// Number of tokens in the prompt. Present only if `done` is `true`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_eval_count: Option<u32>,
    /// Time spent evaluating the prompt (nanoseconds). Present only if `done` is `true`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_eval_duration: Option<u64>,
    /// Number of tokens in the response. Present only if `done` is `true`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eval_count: Option<u32>,
    /// Time spent generating the response (nanoseconds). Present only if `done` is `true`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eval_duration: Option<u64>,
}

/// Streaming `/api/generate` chunks are identical to the existing struct.
pub type GenerateStreamChunk = super::generate::GenerateResponse;
