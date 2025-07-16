use tracing::instrument;

use crate::{models::{invocation::{invocation_handler::{FlowFuture, InvokeFuture}, invocation_util::{call_model, call_tools, generate_llm_request, generate_llm_request_without_tools}}, AgentError}, Agent, Message, Notification};

#[instrument(level = "debug", skip(agent, prompt))]
pub fn simple_loop_invoke<'a>(
    agent: &'a mut Agent,
    prompt: String,
) -> FlowFuture<'a> {
    Box::pin(async move {
    
        agent.history.push(Message::user(prompt));
        
        loop {
            
            let response = invoke(agent).await?;

            if let Some(tc) = response.message.tool_calls {
                for tool_msg in call_tools(agent, &tc).await {
                    agent.history.push(tool_msg);
                }
            } 
            
            else {
                if let Some(stopword) = &agent.stopword {
                    if response
                        .message
                        .content
                        .as_ref()
                        .map_or(false, |c| c.contains(stopword))
                    {
                        agent.notify(Notification::Done(true)).await;
                        return Ok(response.message);
                    } else if let Some(stop_prompt) = &agent.stop_prompt {
                        agent.history.push(Message::tool(stop_prompt, "0"));
                    }
                } else {
                    agent.notify(Notification::Done(true)).await;
                    return Ok(response.message);
                }
            }
        }
    })
}

pub fn invoke<'a>(
    agent: &'a mut Agent,
) -> InvokeFuture<'a> {
    Box::pin(async move {
        let request = generate_llm_request(agent).await?;
        let response = call_model(agent, request).await?;
        Ok(response)
    })
}


pub fn invoke_without_tools<'a>(
    agent: &'a mut Agent,
) -> InvokeFuture<'a> {
    Box::pin(async move {
        let request = generate_llm_request_without_tools(agent).await?;
        let response = call_model(agent, request).await?;
        Ok(response)
    })
}