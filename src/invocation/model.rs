//! Per-call model metadata and context.

use std::marker::PhantomData;

use crate::{services::llm::ClientBuilder, ClientConfig, Notification, Provider};
use tokio::sync::mpsc::Sender;

/// The default invocation output: provider message content remains text.
#[derive(Debug, Clone, Default)]
pub struct Standard;

/// A structured invocation output parsed from assistant message content.
#[derive(Debug, Clone, Default)]
pub struct Structured<T>(pub(crate) PhantomData<T>);

/// One passive, provider-neutral request to a [`crate::Model`].
///
/// An invocation contains only data for a single call. It does not own a client,
/// model identifier, conversation history, or any state that can survive the call.
#[derive(Debug, Clone)]
pub struct Invocation<R, O = Standard> {
    _output: PhantomData<O>,
    pub(crate) request: R,
    pub(crate) model: Option<String>,
    pub(crate) client_config: ClientConfig,
    pub(crate) name: Option<String>,
    pub(crate) notification_channel: Option<Sender<Notification>>,
}

impl<R, O> Invocation<R, O> {
    pub(crate) fn new(request: R) -> Self {
        Self {
            _output: PhantomData,
            request,
            model: None,
            client_config: ClientConfig::default(),
            name: None,
            notification_channel: None,
        }
    }

    pub(crate) fn with_output<T>(self) -> Invocation<R, T> {
        Invocation {
            _output: PhantomData,
            request: self.request,
            model: self.model,
            client_config: self.client_config,
            name: self.name,
            notification_channel: self.notification_channel,
        }
    }

    /// Mark a schema-configured invocation as returning a parsed structured response.
    pub(crate) fn structured_output<T>(self) -> Invocation<R, Structured<T>> {
        self.with_output()
    }

    /// Select the model used by this one-off invocation.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
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

    /// Set the name used for notifications and tracing for this call.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Send events produced by this call to the supplied channel.
    pub fn notification_channel(mut self, channel: Option<Sender<Notification>>) -> Self {
        self.notification_channel = channel;
        self
    }
}
