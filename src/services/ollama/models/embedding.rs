use std::collections::HashMap;

use serde::{Deserialize, Serialize};



#[derive(Serialize, Debug, Clone)]
pub struct EmbeddingsRequest {
    pub model: String,
    pub input: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<HashMap<String, serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<String>,
}

/// Response from the `/api/embeddings` endpoint.
#[derive(Deserialize, Debug, Clone)]
pub struct EmbeddingsResponse {
    pub embedding: Vec<f64>,
}
