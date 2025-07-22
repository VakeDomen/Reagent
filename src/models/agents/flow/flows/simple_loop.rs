use tracing::instrument;

use crate::{models::{flow::{invocation_flows::FlowFuture, util::invocations::{call_tools, invoke}}}, Agent, Message, Notification};

#[instrument(level = "debug", skip(agent, prompt))]
pub fn simple_loop_invoke<'a>(
    agent: &'a mut Agent,
    prompt: String,
) -> FlowFuture<'a> {
    Box::pin(async move {
    
        agent.history.push(Message::user(prompt));
        
        loop {
            
            let response = invoke(agent).await?;
            agent.history.push(response.message.clone());

            if let Some(tc) = response.message.tool_calls {
                for tool_msg in call_tools(agent, &tc).await {
                    agent.history.push(tool_msg);
                }
            } 
            
            else if let Some(stopword) = &agent.stopword {
                if response
                    .message
                    .content
                    .as_ref()
                    .is_some_and(|c| c.contains(stopword))
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
    })
}

