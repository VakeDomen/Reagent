use crate::{ChatInvocation, ChatResponse, InvocationError};

use super::Agent;

impl Agent {
    /// Execute one model call using this agent's notification context.
    ///
    /// This method does not update history. Flows remain responsible for deciding
    /// which request and response messages become agent state.
    pub async fn invoke_model(
        &self,
        mut invocation: ChatInvocation,
    ) -> Result<ChatResponse, InvocationError> {
        if invocation.name.is_none() {
            invocation.name = Some(self.name.clone());
        }
        if invocation.notification_channel.is_none() {
            invocation.notification_channel = self.notification_channel.clone();
        }
        invocation
            .model(self.model.id().to_owned())
            .client_config(self.model.client_config().clone())
            .invoke()
            .await
    }
}
