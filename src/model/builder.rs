use std::{collections::HashMap, marker::PhantomData, path::PathBuf};

use crate::{
    services::llm::{ClientBuilder, ResponseFormatConfig},
    services::systemone::{HasQuestions, NoQuestions, SystemOneClient},
    ClientConfig, Embedding, InferenceOptions, InvocationError, Llm, LoadTemplateError, Message,
    Model, Notification, NoulCriteria, Prompt, Provider, Question, SchemaSpec, Standard,
    Structured, SystemOne, SystemOneQuestion, Template, TemplateDataSource, TemplateInput, Tool,
};
use serde_json::Value;
use tokio::sync::mpsc::{self, Sender};

pub type LlmModelBuilder = ModelBuilder<Llm, Standard>;
pub type EmbeddingModelBuilder = ModelBuilder<Embedding>;
pub type SystemOneModelBuilder<Q = NoQuestions> = ModelBuilder<SystemOne, Standard, Q>;

/// Kinds and input states for which a model is ready to build.
#[doc(hidden)]
pub trait BuildableModel<M> {
    fn validate(kind: &M, id: Option<&str>, config: &ClientConfig) -> Result<(), InvocationError>;
}

/// Model kinds that emit inference notifications.
#[doc(hidden)]
pub trait NotificationCapableModel {}

impl<I> NotificationCapableModel for (Llm, I) {}
impl<I> NotificationCapableModel for (Embedding, I) {}

impl<I> BuildableModel<Llm> for (Llm, I) {
    fn validate(_: &Llm, id: Option<&str>, config: &ClientConfig) -> Result<(), InvocationError> {
        if id.is_none() {
            return Err(InvocationError::ModelNotDefined);
        }
        config.clone().build()?;
        Ok(())
    }
}

impl<I> BuildableModel<Embedding> for (Embedding, I) {
    fn validate(
        _: &Embedding,
        id: Option<&str>,
        config: &ClientConfig,
    ) -> Result<(), InvocationError> {
        if id.is_none() {
            return Err(InvocationError::ModelNotDefined);
        }
        config.clone().build()?;
        Ok(())
    }
}

impl BuildableModel<SystemOne> for (SystemOne, HasQuestions) {
    fn validate(
        kind: &SystemOne,
        _: Option<&str>,
        config: &ClientConfig,
    ) -> Result<(), InvocationError> {
        kind.questions
            .validate()
            .map_err(InvocationError::InvalidSystemOneQuestion)?;
        SystemOneClient::new(config.clone())?;
        Ok(())
    }
}

/// Builds a reusable [`Model`]. Invocation inputs intentionally do not belong
/// here; they are supplied to [`Model::invoke`] for each call.
#[derive(Debug, Clone)]
pub struct ModelBuilder<M = Llm, O = Standard, I = Prompt> {
    id: Option<String>,
    client_config: ClientConfig,
    options: InferenceOptions,
    stream: bool,
    keep_alive: Option<String>,
    history: Vec<Message>,
    tools: Option<Vec<Tool>>,
    response_format: ResponseFormatConfig,
    provider_format: Option<Value>,
    name: Option<String>,
    notification_channel: Option<Sender<Notification>>,
    template: Option<Template>,
    kind: M,
    markers: PhantomData<(O, I)>,
}

impl<M: Default, O, I> Default for ModelBuilder<M, O, I> {
    fn default() -> Self {
        Self {
            id: None,
            client_config: ClientConfig::default(),
            options: InferenceOptions::default(),
            stream: false,
            keep_alive: None,
            history: Vec::new(),
            tools: None,
            response_format: ResponseFormatConfig::default(),
            provider_format: None,
            name: None,
            notification_channel: None,
            template: None,
            kind: M::default(),
            markers: PhantomData,
        }
    }
}

impl<M, O, I> ModelBuilder<M, O, I> {
    pub fn new(id: impl Into<String>) -> Self
    where
        M: Default,
    {
        Self {
            id: Some(id.into()),
            ..Self::default()
        }
    }

    pub fn model(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    pub fn client_config(mut self, config: ClientConfig) -> Self {
        self.client_config = config;
        self
    }

    pub fn provider(mut self, provider: Provider) -> Self {
        self.client_config = self.client_config.provider(Some(provider));
        self
    }

    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.client_config = self.client_config.base_url(Some(base_url));
        self
    }

    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.client_config = self.client_config.api_key(Some(api_key));
        self
    }

    pub fn organization(mut self, organization: impl Into<String>) -> Self {
        self.client_config = self.client_config.organization(Some(organization));
        self
    }

    pub fn extra_headers(mut self, headers: HashMap<String, String>) -> Self {
        self.client_config = self.client_config.extra_headers(Some(headers));
        self
    }

    pub fn build(self) -> Result<Model<M, O, I>, InvocationError>
    where
        (M, I): BuildableModel<M>,
    {
        <(M, I) as BuildableModel<M>>::validate(
            &self.kind,
            self.id.as_deref(),
            &self.client_config,
        )?;

        Ok(Model::new(
            self.id,
            self.client_config,
            self.options,
            self.stream,
            self.keep_alive,
            self.history,
            self.tools,
            self.response_format,
            self.provider_format,
            self.name,
            self.notification_channel,
            self.template,
            self.kind,
        ))
    }

    /// Build a model with a channel for inference notifications.
    ///
    /// The channel receives events emitted while invoking the model, including
    /// streaming token notifications when streaming is enabled.
    pub fn build_with_notification(
        mut self,
    ) -> Result<(Model<M, O, I>, mpsc::Receiver<Notification>), InvocationError>
    where
        (M, I): BuildableModel<M> + NotificationCapableModel,
    {
        let (sender, receiver) = mpsc::channel(100);
        self.notification_channel = Some(sender);
        let model = self.build()?;
        Ok((model, receiver))
    }
}

impl ModelBuilder<SystemOne, Standard, NoQuestions> {
    pub fn systemone() -> Self {
        let mut builder = Self::default();
        builder.client_config.provider = Some(Provider::SystemOne);
        builder
    }
}

impl<Q> ModelBuilder<SystemOne, Standard, Q> {
    /// Add a named question, including an advanced structured question.
    pub fn question(
        mut self,
        key: impl Into<String>,
        question: SystemOneQuestion,
    ) -> ModelBuilder<SystemOne, Standard, HasQuestions> {
        self.kind.questions.insert(key.into(), question);
        ModelBuilder {
            id: self.id,
            client_config: self.client_config,
            options: self.options,
            stream: self.stream,
            keep_alive: self.keep_alive,
            history: self.history,
            tools: self.tools,
            response_format: self.response_format,
            provider_format: self.provider_format,
            name: self.name,
            notification_channel: self.notification_channel,
            template: self.template,
            kind: self.kind,
            markers: PhantomData,
        }
    }

    pub fn noul(
        self,
        key: impl Into<String>,
        instructions: impl Into<Question>,
    ) -> ModelBuilder<SystemOne, Standard, HasQuestions> {
        self.question(key, SystemOneQuestion::noul(instructions))
    }

    pub fn noul_with_criteria(
        self,
        key: impl Into<String>,
        instructions: impl Into<Question>,
        criteria: NoulCriteria,
    ) -> ModelBuilder<SystemOne, Standard, HasQuestions> {
        self.question(
            key,
            SystemOneQuestion::noul_with_criteria(instructions, criteria),
        )
    }

    pub fn choice<K, V, C>(
        self,
        key: impl Into<String>,
        instructions: impl Into<Question>,
        criteria: C,
    ) -> ModelBuilder<SystemOne, Standard, HasQuestions>
    where
        K: Into<String>,
        V: Into<String>,
        C: IntoIterator<Item = (K, V)>,
    {
        self.question(key, SystemOneQuestion::choice(instructions, criteria))
    }

    pub fn score<V, C>(
        self,
        key: impl Into<String>,
        instructions: impl Into<Question>,
        criteria: C,
    ) -> ModelBuilder<SystemOne, Standard, HasQuestions>
    where
        V: Into<String>,
        C: IntoIterator<Item = V>,
    {
        self.question(key, SystemOneQuestion::score(instructions, criteria))
    }
}

impl<O, I> ModelBuilder<Llm, O, I> {
    pub fn keep_alive(mut self, keep_alive: impl Into<String>) -> Self {
        self.keep_alive = Some(keep_alive.into());
        self
    }

    pub fn set_history(mut self, history: impl Into<Vec<Message>>) -> Self {
        self.history = history.into();
        self
    }

    pub fn tools(mut self, tools: Vec<Tool>) -> Self {
        self.tools = Some(tools);
        self
    }

    pub fn add_tool(mut self, tool: Tool) -> Self {
        self.tools.get_or_insert_with(Vec::new).push(tool);
        self
    }

    pub fn remove_tools(mut self) -> Self {
        self.tools = None;
        self
    }

    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn notification_channel(mut self, channel: Option<Sender<Notification>>) -> Self {
        self.notification_channel = channel;
        self
    }

    /// Escape hatch for an already provider-formatted response format.
    pub fn provider_format(mut self, format: Value) -> Self {
        self.provider_format = Some(format);
        self
    }

    pub fn schema_name(mut self, name: impl Into<String>) -> Self {
        self.response_format.set_name(name);
        self
    }

    pub fn schema_strict(mut self, strict: bool) -> Self {
        self.response_format.set_strict(strict);
        self
    }

    pub fn options(mut self, options: InferenceOptions) -> Self {
        self.options = options;
        self
    }

    pub fn stream(mut self, stream: bool) -> Self {
        self.stream = stream;
        self
    }

    pub fn temperature(mut self, value: f32) -> Self {
        self.options.temperature = Some(value);
        self
    }

    pub fn top_p(mut self, value: f32) -> Self {
        self.options.top_p = Some(value);
        self
    }

    pub fn presence_penalty(mut self, value: f32) -> Self {
        self.options.presence_penalty = Some(value);
        self
    }

    pub fn frequency_penalty(mut self, value: f32) -> Self {
        self.options.frequency_penalty = Some(value);
        self
    }

    pub fn num_ctx(mut self, value: u32) -> Self {
        self.options.num_ctx = Some(value);
        self
    }

    pub fn repeat_last_n(mut self, value: i32) -> Self {
        self.options.repeat_last_n = Some(value);
        self
    }

    pub fn repeat_penalty(mut self, value: f32) -> Self {
        self.options.repeat_penalty = Some(value);
        self
    }

    pub fn seed(mut self, value: i32) -> Self {
        self.options.seed = Some(value);
        self
    }

    pub fn stop(mut self, value: impl Into<String>) -> Self {
        self.options.stop = Some(value.into());
        self
    }

    pub fn num_predict(mut self, value: i32) -> Self {
        self.options.num_predict = Some(value);
        self
    }

    pub fn max_tokens(mut self, value: i32) -> Self {
        self.options.max_tokens = Some(value);
        self
    }

    pub fn top_k(mut self, value: u32) -> Self {
        self.options.top_k = Some(value);
        self
    }

    pub fn min_p(mut self, value: f32) -> Self {
        self.options.min_p = Some(value);
        self
    }
}

impl<O, I> ModelBuilder<Embedding, O, I> {
    pub fn keep_alive(mut self, keep_alive: impl Into<String>) -> Self {
        self.keep_alive = Some(keep_alive.into());
        self
    }
}

impl<I> ModelBuilder<Llm, Standard, I> {
    pub fn structured_output<T: rmcp::schemars::JsonSchema>(
        self,
    ) -> ModelBuilder<Llm, Structured<T>, I> {
        ModelBuilder {
            id: self.id,
            client_config: self.client_config,
            options: self.options,
            stream: self.stream,
            keep_alive: self.keep_alive,
            history: self.history,
            tools: self.tools,
            response_format: {
                let mut format = self.response_format;
                format.set_type::<T>();
                format
            },
            provider_format: self.provider_format,
            name: self.name,
            notification_channel: self.notification_channel,
            template: self.template,
            kind: self.kind,
            markers: PhantomData,
        }
    }

    pub fn response_format(
        mut self,
        schema: SchemaSpec,
    ) -> ModelBuilder<Llm, Structured<Value>, I> {
        self.response_format.set_spec(schema);
        self.with_structured_value()
    }

    pub fn response_format_str(mut self, schema: &str) -> ModelBuilder<Llm, Structured<Value>, I> {
        self.response_format.set_raw(schema);
        self.with_structured_value()
    }

    pub fn response_format_value(
        mut self,
        schema: Value,
    ) -> ModelBuilder<Llm, Structured<Value>, I> {
        self.response_format.set_value(schema);
        self.with_structured_value()
    }

    pub fn response_format_from<T: rmcp::schemars::JsonSchema>(
        self,
    ) -> ModelBuilder<Llm, Structured<T>, I> {
        self.structured_output()
    }

    fn with_structured_value(self) -> ModelBuilder<Llm, Structured<Value>, I> {
        ModelBuilder {
            id: self.id,
            client_config: self.client_config,
            options: self.options,
            stream: self.stream,
            keep_alive: self.keep_alive,
            history: self.history,
            tools: self.tools,
            response_format: self.response_format,
            provider_format: self.provider_format,
            name: self.name,
            notification_channel: self.notification_channel,
            template: self.template,
            kind: self.kind,
            markers: PhantomData,
        }
    }
}

impl<I, O> ModelBuilder<Llm, Structured<O>, I> {
    pub fn response_format(mut self, schema: SchemaSpec) -> Self {
        self.response_format.set_spec(schema);
        self
    }

    pub fn response_format_str(mut self, schema: &str) -> Self {
        self.response_format.set_raw(schema);
        self
    }

    pub fn response_format_value(mut self, schema: Value) -> Self {
        self.response_format.set_value(schema);
        self
    }
}

impl<O> ModelBuilder<Llm, O, Prompt> {
    /// Configure this model to render a template for each invocation.
    ///
    /// The resulting model accepts a `HashMap` of template data in
    /// [`Model::invoke`]. Rendering is ephemeral and does not alter model
    /// history between calls.
    pub fn set_template(self, template: Template) -> ModelBuilder<Llm, O, TemplateInput> {
        ModelBuilder {
            id: self.id,
            client_config: self.client_config,
            options: self.options,
            stream: self.stream,
            keep_alive: self.keep_alive,
            history: self.history,
            tools: self.tools,
            response_format: self.response_format,
            provider_format: self.provider_format,
            name: self.name,
            notification_channel: self.notification_channel,
            template: Some(template),
            kind: self.kind,
            markers: PhantomData,
        }
    }

    pub fn set_template_simple(
        self,
        content: impl Into<String>,
    ) -> ModelBuilder<Llm, O, TemplateInput> {
        self.set_template(Template::simple(content))
    }

    pub fn set_template_with_source<D>(
        self,
        content: &str,
        data_source: D,
    ) -> ModelBuilder<Llm, O, TemplateInput>
    where
        D: TemplateDataSource + 'static,
    {
        self.set_template(Template::new(content, data_source))
    }

    pub fn set_template_from_file<P>(
        self,
        path: P,
    ) -> Result<ModelBuilder<Llm, O, TemplateInput>, LoadTemplateError>
    where
        P: Into<PathBuf>,
    {
        Ok(self.set_template(Template::from_file(path)?))
    }

    pub fn set_template_from_file_with_source<P, D>(
        self,
        path: P,
        data_source: D,
    ) -> Result<ModelBuilder<Llm, O, TemplateInput>, LoadTemplateError>
    where
        P: Into<PathBuf>,
        D: TemplateDataSource + 'static,
    {
        Ok(self.set_template(Template::from_file_with_source(path, data_source)?))
    }
}
