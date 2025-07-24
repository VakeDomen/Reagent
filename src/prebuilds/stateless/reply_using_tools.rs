use crate::{models::agents::flow::invocation_flows::{Flow, FlowFuture}, prebuilds::stateless::StatelessPrebuild, util::invocations::{call_tools, invoke}, Agent, AgentBuilder, Message};


fn custom_flow<'a>(agent: &'a mut Agent, prompt: String) -> FlowFuture<'a> {
    Box::pin(async move {
        agent.clear_history();
        agent.history.push(Message::system(agent.system_prompt.clone()));
        agent.history.push(Message::user(prompt));
        let response = invoke(agent).await?;
        if let Some(tc) = response.message.tool_calls.clone() {
            for tool_msg in call_tools(agent, &tc).await {
                agent.history.push(tool_msg);
            }
        } 
        agent.notify(crate::NotificationContent::Done(true)).await;
        Ok(response.message)
    })    
}

impl StatelessPrebuild {
    pub fn reply_using_tools() -> AgentBuilder {
        AgentBuilder::default()
            .set_flow(Flow::Custom(custom_flow))
            .set_name("Stateless_prebuild-reply_using_tools")
    }
}