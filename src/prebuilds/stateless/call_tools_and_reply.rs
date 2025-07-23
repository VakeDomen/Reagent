use crate::{models::agents::flow::invocation_flows::{Flow, FlowFuture}, util::invocations::{call_tools, invoke}, Agent, AgentBuilder, Message};


fn custom_flow<'a>(agent: &'a mut Agent, prompt: String) -> FlowFuture<'a> {
    Box::pin(async move {
        agent.clear_history();
        agent.history.push(Message::system(agent.system_prompt.clone()));
        agent.history.push(Message::user(prompt));
        let mut response = invoke(agent).await?;
        if let Some(tc) = response.message.tool_calls {
            for tool_msg in call_tools(agent, &tc).await {
                agent.history.push(tool_msg);
            }
            response = invoke(agent).await?;
        } 
        Ok(response.message)
    })    
}

impl AgentBuilder {
    pub fn call_tools_and_reply() -> AgentBuilder {
        AgentBuilder::default()
            .set_flow(Flow::Custom(custom_flow))
            .set_name("Stateless_prebuild-call_tools_and_reply")
    }
}