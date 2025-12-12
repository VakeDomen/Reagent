use tokio::sync::mpsc::Sender;

use crate::{Notification, NotificationHandler};

pub struct NotificationOutputChannel {
    sender: Option<Sender<Notification>>,
    name: String,
}

impl NotificationOutputChannel {
    pub fn new(sender: Option<Sender<Notification>>, name: String) -> Self {
        Self { sender, name }
    }
}

impl NotificationHandler for NotificationOutputChannel {
    fn get_outgoing_channel(&self) -> &Option<Sender<Notification>> {
        &self.sender
    }

    fn get_channel_name(&self) -> &String {
        &self.name
    }
}
