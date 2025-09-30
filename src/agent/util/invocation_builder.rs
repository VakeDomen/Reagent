use serde_json::Value;

use crate::{
    call_tools,
    services::llm::{BaseRequest, InferenceOptions},
    Agent, AgentError, ChatRequest, ChatResponse, Message, Tool,
};

#[derive(Debug, Clone)]
enum InheritOpt<T> {
    Inherit,
    Value(T),
}

#[derive(Debug, Clone)]
enum InheritMaybe<T> {
    Inherit,
    Some(T),
    ExplicitNone,
}

impl<T> Default for InheritOpt<T> {
    fn default() -> Self {
        InheritOpt::Inherit
    }
}
impl<T> Default for InheritMaybe<T> {
    fn default() -> Self {
        InheritMaybe::Inherit
    }
}

#[derive(Debug, Clone, Default)]
pub struct InvocationBuilder {
    model: InheritOpt<String>,
    format: InheritMaybe<Value>,
    stream: InheritMaybe<bool>,
    keep_alive: InheritMaybe<String>,

    // payload
    messages: InheritOpt<Vec<Message>>,
    tools: InheritMaybe<Vec<Tool>>,

    // flattened options: None means inherit, Some(_) means override
    opts: InferenceOptions,
    // strip_thinking: In,
}

impl InvocationBuilder {
    pub fn model(mut self, v: impl Into<String>) -> Self {
        self.model = InheritOpt::Value(v.into());
        self
    }
    pub fn response_format_some(mut self, v: Value) -> Self {
        self.format = InheritMaybe::Some(v);
        self
    }
    pub fn response_format_none(mut self) -> Self {
        self.format = InheritMaybe::ExplicitNone;
        self
    }
    pub fn stream_some(mut self, v: bool) -> Self {
        self.stream = InheritMaybe::Some(v);
        self
    }
    pub fn stream_none(mut self) -> Self {
        self.stream = InheritMaybe::ExplicitNone;
        self
    }
    pub fn keep_alive_some(mut self, v: impl Into<String>) -> Self {
        self.keep_alive = InheritMaybe::Some(v.into());
        self
    }
    pub fn keep_alive_none(mut self) -> Self {
        self.keep_alive = InheritMaybe::ExplicitNone;
        self
    }
    pub fn messages(mut self, msgs: Vec<Message>) -> Self {
        self.messages = InheritOpt::Value(msgs);
        self
    }
    pub fn add_message(mut self, msg: Message) -> Self {
        match &mut self.messages {
            InheritOpt::Value(v) => v.push(msg),
            InheritOpt::Inherit => self.messages = InheritOpt::Value(vec![msg]),
        }
        self
    }
    pub fn tools_some(mut self, tools: Vec<Tool>) -> Self {
        self.tools = InheritMaybe::Some(tools);
        self
    }
    pub fn tools_none(mut self) -> Self {
        self.tools = InheritMaybe::ExplicitNone;
        self
    }
    pub fn add_tool(mut self, tool: Tool) -> Self {
        match &mut self.tools {
            InheritMaybe::Some(v) => v.push(tool),
            InheritMaybe::ExplicitNone => { /* explicit none, do nothing */ }
            InheritMaybe::Inherit => self.tools = InheritMaybe::Some(vec![tool]),
        }
        self
    }

    // flattened inference options setters
    pub fn num_ctx(mut self, v: Option<i32>) -> Self {
        self.opts.num_ctx = v;
        self
    }
    pub fn repeat_last_n(mut self, v: Option<i32>) -> Self {
        self.opts.repeat_last_n = v;
        self
    }
    pub fn repeat_penalty(mut self, v: Option<f32>) -> Self {
        self.opts.repeat_penalty = v;
        self
    }
    pub fn temperature(mut self, v: Option<f32>) -> Self {
        self.opts.temperature = v;
        self
    }
    pub fn seed(mut self, v: Option<i32>) -> Self {
        self.opts.seed = v;
        self
    }
    pub fn stop(mut self, v: Option<Vec<String>>) -> Self {
        self.opts.stop = v;
        self
    }
    pub fn num_predict(mut self, v: Option<i32>) -> Self {
        self.opts.num_predict = v;
        self
    }
    pub fn top_k(mut self, v: Option<i32>) -> Self {
        self.opts.top_k = v;
        self
    }
    pub fn top_p(mut self, v: Option<f32>) -> Self {
        self.opts.top_p = v;
        self
    }
    pub fn min_p(mut self, v: Option<f32>) -> Self {
        self.opts.min_p = v;
        self
    }
    pub fn presence_penalty(mut self, v: Option<f32>) -> Self {
        self.opts.presence_penalty = v;
        self
    }
    pub fn frequency_penalty(mut self, v: Option<f32>) -> Self {
        self.opts.frequency_penalty = v;
        self
    }
    pub fn max_tokens(mut self, v: Option<i32>) -> Self {
        self.opts.max_tokens = v;
        self
    }

    pub fn strip_thinking(mut self, strip_thinking: bool) -> Self {
        self.strip_thinking = Some(strip_thinking);
        self
    }

    pub async fn invoke(self, agent: &mut Agent) -> Result<ChatResponse, AgentError> {
        let model = match self.model {
            InheritOpt::Value(v) => v,
            InheritOpt::Inherit => agent.model.clone(),
        };

        let format = match self.format {
            InheritMaybe::Some(v) => Some(v),
            InheritMaybe::ExplicitNone => None,
            InheritMaybe::Inherit => agent.response_format.clone(),
        };

        let stream = match self.stream {
            InheritMaybe::Some(v) => Some(v),
            InheritMaybe::ExplicitNone => None,
            InheritMaybe::Inherit => Some(agent.stream),
        };

        let keep_alive = match self.keep_alive {
            InheritMaybe::Some(v) => Some(v),
            InheritMaybe::ExplicitNone => None,
            InheritMaybe::Inherit => None, // no agent value, so omit unless set
        };

        // merge messages and tools
        let messages = match self.messages {
            InheritOpt::Value(v) => v,
            InheritOpt::Inherit => agent.history.clone(),
        };

        let tools = match self.tools {
            InheritMaybe::Some(v) => Some(v),
            InheritMaybe::ExplicitNone => None,
            InheritMaybe::Inherit => agent.tools.clone(),
        };

        // merge inference options field by field
        let merged_opts = InferenceOptions {
            num_ctx: self.opts.num_ctx.or(agent.num_ctx),
            repeat_last_n: self.opts.repeat_last_n.or(agent.repeat_last_n),
            repeat_penalty: self.opts.repeat_penalty.or(agent.repeat_penalty),
            temperature: self.opts.temperature.or(agent.temperature),
            seed: self.opts.seed.or(agent.seed),
            stop: self.opts.stop.or_else(|| agent.stop.clone()),
            num_predict: self.opts.num_predict.or(agent.num_predict),
            top_k: self.opts.top_k.or(agent.top_k),
            top_p: self.opts.top_p.or(agent.top_p),
            min_p: self.opts.min_p.or(agent.min_p),
            presence_penalty: self.opts.presence_penalty.or(agent.presence_penalty),
            frequency_penalty: self.opts.frequency_penalty.or(agent.frequency_penalty),
            max_tokens: self.opts.max_tokens.or(None),
        };

        let options = if all_none(&merged_opts) {
            None
        } else {
            Some(merged_opts)
        };

        let request = ChatRequest {
            base: BaseRequest {
                model,
                format,
                options,
                stream,
                keep_alive,
            },
            messages,
            tools,
        };

        // let mut request: ChatRequest = (&*agent).into();

        // if self.stream.is_some() {
        //     request.base.stream = self.stream;
        // }

        // if let Some(use_tools) = self.use_tools {
        //     match use_tools {
        //         true => request.tools = agent.tools.clone(),
        //         false => request.tools = None,
        //     }
        // }

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

fn all_none(o: &InferenceOptions) -> bool {
    o.num_ctx.is_none()
        && o.repeat_last_n.is_none()
        && o.repeat_penalty.is_none()
        && o.temperature.is_none()
        && o.seed.is_none()
        && o.stop.is_none()
        && o.num_predict.is_none()
        && o.top_k.is_none()
        && o.top_p.is_none()
        && o.min_p.is_none()
        && o.presence_penalty.is_none()
        && o.frequency_penalty.is_none()
        && o.max_tokens.is_none()
}
