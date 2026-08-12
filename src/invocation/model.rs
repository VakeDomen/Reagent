//! Per-call model metadata and context.

use crate::Notification;
use tokio::sync::mpsc::Sender;

/// One passive, provider-neutral request to a [`crate::Model`].
///
/// An invocation contains only data for a single call. It does not own a client,
/// model identifier, conversation history, or any state that can survive the call.
#[derive(Debug, Clone)]
pub struct Invocation<R> {
    pub(crate) request: R,
    pub(crate) name: Option<String>,
    pub(crate) notification_channel: Option<Sender<Notification>>,
}

impl<R> Invocation<R> {
    pub(crate) fn new(request: R) -> Self {
        Self {
            request,
            name: None,
            notification_channel: None,
        }
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
