use tokio::sync::mpsc::Sender;

use crate::{services::llm::InferenceClient, ChatRequest, Notification, OutChannel};

pub struct InvocationRequest {
    pub strip_thinking: bool,
    pub request: ChatRequest,
    pub client: InferenceClient,
    pub notification_channel: OutChannel,
}

impl InvocationRequest {
    pub fn new(
        strip_thinking: bool,
        request: ChatRequest,
        client: InferenceClient,
        notification_channel: Option<Sender<Notification>>,
        name: String,
    ) -> Self {
        let notification_channel = OutChannel::new(notification_channel, name);
        Self {
            strip_thinking,
            request,
            client,
            notification_channel,
        }
    }
}
