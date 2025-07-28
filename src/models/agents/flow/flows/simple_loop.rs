use tracing::instrument;

use crate::{
    models::{agents::flow::invocation_flows::FlowFuture, notification::NotificationContent}, util::invocations::{invoke_with_tool_calls, invoke_without_tools}, Agent, Message, Notification
};

#[instrument(level = "debug", skip(agent, prompt))]
pub fn simple_loop_invoke<'a>(
    agent: &'a mut Agent,
    prompt: String,
) -> FlowFuture<'a> {
    Box::pin(async move {
    
        agent.history.push(Message::user(prompt.clone()));
        let mut iteration_number = 1;

        loop {
            
            let response = invoke_with_tool_calls(agent).await?;
            
            // stop conditions
            let mut done = false;
            if let Some(keyword) = &agent.stopword {
                if response
                    .message
                    .content
                    .as_ref()
                    .is_some_and(|c| c.contains(keyword)) {
                        done = true;
                }
            }

            if let Some(max_iterations) = agent.max_iterations {
                if max_iterations > iteration_number {
                    done = true;
                }
            }

            if done {
                break;
            }

            iteration_number += 1;
        }

        agent.history.push(Message::user(format!("Based on conversation answer the prompt to the best of your ability: {}", prompt)));
        let response = invoke_without_tools(agent).await?;
        agent.history.push(response.message.clone());
        agent.notify(NotificationContent::Done(true, response.message.content.clone())).await;
        return Ok(response.message);
    })
}

