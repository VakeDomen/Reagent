use futures::{stream::SelectAll, StreamExt};
use serde_json::Value;
use tokio::sync::mpsc::{Receiver, Sender};
use tokio_stream::wrappers::ReceiverStream;

use crate::{ChatRequest, ChatResponse, Notification, NotificationContent, Response, Success, Token, ToolCall};


pub trait NotificationHandler {
    fn get_outgoing_channel(&self) -> &Option<Sender<Notification>>;
    fn get_channel_name(&self) -> &String;


    /// Send a notification with the given content.
    ///
    /// Returns `true` if successfully delivered, `false` otherwise.
    async fn notify(&self, content: NotificationContent) -> bool {
        if self.get_outgoing_channel().is_none() {
            return false;
        }
        let notification_channel = self.get_outgoing_channel()
            .as_ref()
            .unwrap();

        match notification_channel.send(Notification::new( 
            self.get_channel_name().clone(), 
            content, 
        )).await {
            Ok(_) => true,
            Err(e) => {
                tracing::error!(error = %e, "Failed sending notification");
                false
            },
        }
    }

    /// Forward notifications from an external receiver into this agent’s notification 
    /// output channel.
    fn forward_notifications(
        &self,
        mut from_channel: Receiver<Notification>
    ) {
        if let Some(notification_channel) = &self.get_outgoing_channel() {
            let to_sender = notification_channel.clone();
            tokio::spawn(async move {
                while let Some(msg) = from_channel.recv().await {

                    if to_sender.send(msg.unwrap()).await.is_err() {
                        break;
                    }
                }
            });    
        }
    }

    /// Merge any number of `Receiver<Notification>` streams into one,
    /// and forward all messages into this agent’s notification output channel.
    fn forward_multiple_notifications<I>(&self, channels: I)
    where
        I: IntoIterator<Item = Receiver<Notification>>,
    {
        let to_sender = match &self.get_outgoing_channel() {
            Some(s) => s.clone(),
            None => return,
        };

        let mut merged = SelectAll::new();
        for rx in channels {
            let stream = ReceiverStream::new(rx)
                .map(|notif| notif);
            merged.push(stream);
        }

        tokio::spawn(async move {
            while let Some(notification) = merged.next().await {
                if to_sender.send(notification).await.is_err() {
                    break;
                }
            }
        });
    }

    async fn notify_done(&self, success: Success, resp: Response) -> bool {
        self.notify(NotificationContent::Done(success, resp)).await
    }
    async fn notify_prompt_request(&self, req: ChatRequest) -> bool {
        self.notify(NotificationContent::PromptRequest(req)).await
    }
    async fn notify_poompt_success(&self, resp: ChatResponse) -> bool {
        self.notify(NotificationContent::PromptSuccessResult(resp)).await
    }
    async fn notify_poompt_error(&self, error_message: String) -> bool {
        self.notify(NotificationContent::PromptErrorResult(error_message)).await
    }
    async fn notify_tool_request(&self, tool_call: ToolCall) -> bool {
        self.notify(NotificationContent::ToolCallRequest(tool_call)).await
    }
    async fn notify_tool_success(&self, tool_result: String) -> bool {
        self.notify(NotificationContent::ToolCallSuccessResult(tool_result)).await
    }
    async fn notify_tool_error(&self, error_message: String) -> bool {
        self.notify(NotificationContent::ToolCallErrorResult(error_message)).await
    }
    async fn notify_token(&self, token: Token) -> bool {
        self.notify(NotificationContent::Token(token)).await
    }
    async fn notify_mcp_tool_notification(&self, notification: String) -> bool {
        self.notify(NotificationContent::McpToolNotification(notification)).await
    }
    async fn notify_custom(&self, custom_val: Value) -> bool {
        self.notify(NotificationContent::Custom(custom_val)).await
    }
}