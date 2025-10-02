use std::collections::HashMap;

use crate::{
    services::llm::{InferenceClient, InferenceClientError},
    Provider,
};

#[derive(Debug, Clone, Default)]
pub struct ClientConfig {
    pub provider: Option<Provider>,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub organization: Option<String>,
    pub extra_headers: Option<std::collections::HashMap<String, String>>,
}

pub trait ClientBuilder {
    fn provider(self, provider: Option<Provider>) -> Self;
    fn base_url(self, base_url: Option<impl Into<String>>) -> Self;
    fn api_key(self, api_key: Option<impl Into<String>>) -> Self;
    fn organization(self, organization: Option<impl Into<String>>) -> Self;
    fn extra_headers(self, extra_headers: Option<HashMap<String, String>>) -> Self;
    fn build(self) -> Result<InferenceClient, InferenceClientError>;
}

impl ClientBuilder for ClientConfig {
    fn provider(mut self, provider: Option<Provider>) -> Self {
        self.provider = provider;
        self
    }

    fn base_url(mut self, base_url: Option<impl Into<String>>) -> Self {
        self.base_url = base_url.map(|s| s.into());
        self
    }

    fn api_key(mut self, api_key: Option<impl Into<String>>) -> Self {
        self.api_key = api_key.map(|s| s.into());
        self
    }

    fn organization(mut self, organization: Option<impl Into<String>>) -> Self {
        self.organization = organization.map(|s| s.into());
        self
    }

    fn extra_headers(mut self, extra_headers: Option<HashMap<String, String>>) -> Self {
        self.extra_headers = extra_headers;
        self
    }

    fn build(self) -> Result<InferenceClient, InferenceClientError> {
        InferenceClient::try_from(ClientConfig {
            provider: self.provider.or(Some(Provider::Ollama)),
            base_url: self.base_url,
            api_key: self.api_key,
            organization: self.organization,
            extra_headers: self.extra_headers,
        })
    }
}
