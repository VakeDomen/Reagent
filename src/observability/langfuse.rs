use opentelemetry::{global, trace::TracerProvider as _, KeyValue};
use opentelemetry_langfuse::ExporterBuilder;
use opentelemetry_sdk::trace::span_processor_with_async_runtime::BatchSpanProcessor;
use opentelemetry_sdk::{resource::Resource, runtime, trace::SdkTracerProvider};
use opentelemetry_semantic_conventions::resource::{SERVICE_NAME, SERVICE_VERSION};
use tracing::{Metadata, Subscriber};
use tracing_subscriber::fmt::time::UtcTime;
use tracing_subscriber::layer::Filter;
use tracing_subscriber::prelude::*; // Import prelude for .with()
use tracing_subscriber::{fmt, EnvFilter, Registry};

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
        let name = meta.name();

        // Drop noisy spans by name
        if name == "serve_inner" || name == "streamable_http_session" {
            return false;
        }

        // Allow only your crate (whitelist)
        target.starts_with("reagent_rs")
    }
}

pub fn init(config: LangfuseOptions) -> SdkTracerProvider {
    // 1. Setup Langfuse Exporter
    let mut builder = ExporterBuilder::default();
    if let (Some(pk), Some(sk)) = (config.public_key, config.secret_key) {
        builder = builder.with_basic_auth(pk, sk);
    }
    if let Some(host) = config.host {
        builder = builder.with_host(host);
    }
    let exporter = builder.build().expect("Failed to build Langfuse exporter");

    // 2. Setup Resource
    let resource = Resource::builder()
        .with_attributes([
            KeyValue::new(SERVICE_NAME, "reagent-rs"),
            KeyValue::new(SERVICE_VERSION, env!("CARGO_PKG_VERSION")),
        ])
        .build();

    // 3. Setup Provider
    let processor = BatchSpanProcessor::builder(exporter, runtime::Tokio).build();
    let provider = SdkTracerProvider::builder()
        .with_resource(resource)
        .with_span_processor(processor)
        .build();

    let tracer = provider.tracer("reagent-rs");
    global::set_tracer_provider(provider.clone());

    // 4. Create Filters
    // This filter is for the Console/Stdout log
    let console_filter =
        EnvFilter::try_from_env("RUST_LOG").unwrap_or_else(|_| EnvFilter::new("none,reagent=info"));

    // This filter is for Langfuse (whitelist only your crate)
    let otel_filter = RmcpSpanFilter;

    // 5. Create Layers (with attached filters)

    // Layer 1: Console (Filtered by RUST_LOG)
    let fmt_layer = fmt::layer()
        .with_timer(UtcTime::rfc_3339())
        .with_target(false)
        .with_thread_ids(true)
        .with_filter(console_filter); // Filter applied HERE. Consumes console_filter.

    // Layer 2: Langfuse (Filtered by RmcpSpanFilter)
    let otel_layer = tracing_opentelemetry::layer()
        .with_tracer(tracer)
        .with_filter(otel_filter); // Filter applied HERE.

    // 6. Initialize Registry
    // You just add the ALREADY FILTERED layers.
    let _ = Registry::default()
        .with(fmt_layer)
        .with(otel_layer)
        .try_init();

    provider
}
