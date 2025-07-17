use crate::{models::AgentError, services::ollama::models::{base::{BaseRequest, OllamaOptions}, chat::ChatRequest}, Agent};

/// Builds a [`ChatRequest`] from the agent’s state, *including* whatever
/// `agent.tools` currently holds (calling `agent.get_compiled_tools()` if None).  
///
/// # Errors  
/// Returns [`AgentError`] if `get_compiled_tools()` fails.  

pub async fn generate_llm_request(
    agent: &mut Agent
) -> Result<ChatRequest, AgentError> {
    if let None = agent.tools {
        agent.tools = agent.get_compiled_tools().await?;
    }

    Ok(ChatRequest {
        base: BaseRequest {
            model: agent.model.clone(),
            format: agent.response_format.clone(),
            options: Some(OllamaOptions {
                num_ctx: agent.num_ctx,
                repeat_last_n: agent.repeat_last_n,
                repeat_penalty: agent.repeat_penalty,
                temperature: agent.temperature,
                seed: agent.seed,
                stop: agent.stop.clone(),
                num_predict: agent.num_predict,
                top_k: agent.top_k,
                top_p: agent.top_p,
                min_p: agent.min_p,
                presence_penalty: agent.presence_penalty,
                frequency_penalty: agent.frequency_penalty,
            }),
            stream: Some(false),
            keep_alive: Some("5m".to_string()),
        },
        messages: agent.history.clone(),
        tools: agent.tools.clone(),
    })
}

/// Like [`generate_llm_request`] but always sets `tools: None` in the request.
pub async fn generate_llm_request_without_tools(
    agent: &mut Agent
) -> Result<ChatRequest, AgentError> {
    if let None = agent.tools {
        agent.tools = agent.get_compiled_tools().await?;
    }

    Ok(ChatRequest {
        base: BaseRequest {
            model: agent.model.clone(),
            format: agent.response_format.clone(),
            options: Some(OllamaOptions {
                num_ctx: agent.num_ctx,
                repeat_last_n: agent.repeat_last_n,
                repeat_penalty: agent.repeat_penalty,
                temperature: agent.temperature,
                seed: agent.seed,
                stop: agent.stop.clone(),
                num_predict: agent.num_predict,
                top_k: agent.top_k,
                top_p: agent.top_p,
                min_p: agent.min_p,
                presence_penalty: agent.presence_penalty,
                frequency_penalty: agent.frequency_penalty,
            }),
            stream: Some(false),
            keep_alive: Some("5m".to_string()),
        },
        messages: agent.history.clone(),
        tools: None,
    })
}

