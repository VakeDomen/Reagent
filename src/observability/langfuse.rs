use opentelemetry::{global, trace::TracerProvider as _, KeyValue};
use opentelemetry_langfuse::ExporterBuilder;
use opentelemetry_sdk::{
    propagation::TraceContextPropagator, resource::Resource, trace::SdkTracerProvider,
};
use opentelemetry_semantic_conventions::resource::{SERVICE_NAME, SERVICE_VERSION};
use std::env;
use tracing::{Metadata, Subscriber};
use tracing_subscriber::{
    fmt::{self, format::FmtSpan, time::UtcTime},
    layer::{Filter, SubscriberExt},
    util::SubscriberInitExt,
    EnvFilter, Layer,
};

pub struct LangfuseOptions<'a> {
    pub public_key: Option<&'a str>,
    pub secret_key: Option<&'a str>,
    pub host: Option<&'a str>,
}

pub fn init(config: LangfuseOptions) -> SdkTracerProvider {
    // 1. Build the Exporter
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
    // 2. Build the Provider using the high-level batch exporter method
    // This automatically creates the BatchSpanProcessor and binds it to the Tokio runtime
    // Build the tracer provider with batch processing
    let provider = SdkTracerProvider::builder()
        .with_resource(resource)
        .with_simple_exporter(exporter)
        .build();

    let tracer = provider.tracer("reagent-rs");

    // Install the provider as global so other crates use it
    global::set_tracer_provider(provider.clone());

    // Forward tracing events (including Tokio internal spans when enabled) to OTEL
    // and keep console logging with env-based filtering.
    let env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info,tokio=info"));

    // Apply the filter to the OpenTelemetry layer to exclude unwanted rmcp spans
    let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);

    let fmt_layer = fmt::layer()
        .with_timer(UtcTime::rfc_3339())
        .with_thread_ids(true)
        .with_thread_names(true)
        .with_target(true)
        .with_span_events(FmtSpan::ENTER | FmtSpan::EXIT | FmtSpan::CLOSE);

    tracing_subscriber::registry()
        .with(env_filter)
        .with(fmt_layer)
        .with(otel_layer)
        .init();

    provider
}
