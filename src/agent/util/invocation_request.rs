use tokio::sync::mpsc::Sender;

use crate::{services::llm::InferenceClient, ChatRequest, Notification, NotificationOutputChannel};

pub struct InvocationRequest {
    pub strip_thinking: bool,
    pub request: ChatRequest,
    pub client: InferenceClient,
    pub notification_channel: NotificationOutputChannel,
}

impl InvocationRequest {
    pub fn new(
        strip_thinking: bool,
        request: ChatRequest,
        client: InferenceClient,
        notification_channel: Option<Sender<Notification>>,
        name: String,
    ) -> Self {
        let notification_channel = NotificationOutputChannel::new(notification_channel, name);
        Self {
            strip_thinking,
            request,
            client,
            notification_channel,
        }
    }
}
