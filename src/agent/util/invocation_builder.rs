use crate::{call_tools, Agent, AgentError, ChatRequest, ChatResponse};

#[derive(Debug, Clone, Default)]
pub struct InvocationBuilder {
    stream: Option<bool>,
    use_tools: Option<bool>,
    strip_thinking: Option<bool>,
}

impl InvocationBuilder {
    pub fn stream(mut self, stream: bool) -> Self {
        self.stream = Some(stream);
        self
    }

    pub fn use_tools(mut self, use_tools: bool) -> Self {
        self.use_tools = Some(use_tools);
        self
    }

    pub fn strip_thinking(mut self, strip_thinking: bool) -> Self {
        self.strip_thinking = Some(strip_thinking);
        self
    }

    pub async fn invoke(self, agent: &mut Agent) -> Result<ChatResponse, AgentError> {
        let mut request: ChatRequest = (&*agent).into();

        if self.stream.is_some() {
            request.base.stream = self.stream;
        }

        if let Some(use_tools) = self.use_tools {
            match use_tools {
                true => request.tools = agent.tools.clone(),
                false => request.tools = None,
            }
        }

        let response = match &request.base.stream {
            Some(true) => super::invocations::call_model_streaming(agent, request).await?,
            _ => super::invocations::call_model_nonstreaming(agent, request).await?,
        };

        agent.history.push(response.message.clone());

        if let Some(tc) = response.message.tool_calls.clone() {
            for tool_msg in call_tools(agent, &tc).await {
                agent.history.push(tool_msg);
            }
        }

        Ok(response)
    }
}
