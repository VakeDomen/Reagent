use opentelemetry::{global, trace::TracerProvider as _, KeyValue};
use opentelemetry_langfuse::ExporterBuilder;
use opentelemetry_sdk::trace::span_processor_with_async_runtime::BatchSpanProcessor;
use opentelemetry_sdk::{resource::Resource, runtime, trace::SdkTracerProvider};
use opentelemetry_semantic_conventions::resource::{SERVICE_NAME, SERVICE_VERSION};
use tracing::{Metadata, Subscriber};
use tracing_subscriber::fmt::time::UtcTime;
use tracing_subscriber::layer::Filter;
use tracing_subscriber::prelude::__tracing_subscriber_SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::Layer;
use tracing_subscriber::{fmt, Registry};

pub struct LangfuseOptions<'a> {
    pub public_key: Option<&'a str>,
    pub secret_key: Option<&'a str>,
    pub host: Option<&'a str>,
}

#[derive(Debug, Clone)]
struct RmcpSpanFilter;

impl<S> Filter<S> for RmcpSpanFilter
where
    S: Subscriber,
{
    fn enabled(
        &self,
        meta: &Metadata<'_>,
        _cx: &tracing_subscriber::layer::Context<'_, S>,
    ) -> bool {
        let target = meta.target();

        if target.starts_with("reagent_rs") || target.starts_with("rmcp") {
            // Keep your existing specific logic to filter NOISY internal rmcp spans
            let name = meta.name();
            if target.starts_with("rmcp")
                && (name == "serve_inner" || name == "streamable_http_session")
            {
                return false;
            }

            return true;
        }

        // BLOCK EVERYTHING ELSE
        false
    }
}
pub fn init(config: LangfuseOptions) -> SdkTracerProvider {
    let mut builder = ExporterBuilder::default();
    if let (Some(pk), Some(sk)) = (config.public_key, config.secret_key) {
        builder = builder.with_basic_auth(pk, sk);
    }
    if let Some(host) = config.host {
        builder = builder.with_host(host);
    }
    let exporter = builder.build().expect("Failed to build Langfuse exporter");

    let resource = Resource::builder()
        .with_attributes([
            KeyValue::new(SERVICE_NAME, "reagent-rs"),
            KeyValue::new(SERVICE_VERSION, env!("CARGO_PKG_VERSION")),
        ])
        .build();

    let processor = BatchSpanProcessor::builder(exporter, runtime::Tokio).build();

    let provider = SdkTracerProvider::builder()
        .with_resource(resource)
        .with_span_processor(processor)
        .build();

    let tracer = provider.tracer("reagent-rs");
    global::set_tracer_provider(provider.clone());

    let console_filter =
        EnvFilter::try_from_env("RUST_LOG").unwrap_or_else(|_| EnvFilter::new("reagent=info,info"));

    let fmt_layer = fmt::layer()
        .with_timer(UtcTime::rfc_3339())
        .with_target(false)
        .with_thread_ids(true)
        .with_filter(console_filter);

    let otel_filter = RmcpSpanFilter; // Your allow-list filter

    let otel_layer = tracing_opentelemetry::layer()
        .with_tracer(tracer)
        .with_filter(otel_filter);

    Registry::default().with(fmt_layer).with(otel_layer).init();

    provider
}
