//! Per-call model metadata and context.

use crate::{services::llm::ClientBuilder, ClientConfig, Notification, Provider};
use tokio::sync::mpsc::Sender;

/// One passive, provider-neutral request to a [`crate::Model`].
///
/// An invocation contains only data for a single call. It does not own a client,
/// model identifier, conversation history, or any state that can survive the call.
#[derive(Debug, Clone)]
pub struct Invocation<R> {
    pub(crate) request: R,
    pub(crate) model: Option<String>,
    pub(crate) client_config: ClientConfig,
    pub(crate) name: Option<String>,
    pub(crate) notification_channel: Option<Sender<Notification>>,
}

impl<R> Invocation<R> {
    pub(crate) fn new(request: R) -> Self {
        Self {
            request,
            model: None,
            client_config: ClientConfig::default(),
            name: None,
            notification_channel: None,
        }
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
