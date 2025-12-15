use opentelemetry::{global, trace::TracerProvider as _, KeyValue};
use opentelemetry_langfuse::ExporterBuilder;
use opentelemetry_sdk::trace::span_processor_with_async_runtime::BatchSpanProcessor;
use opentelemetry_sdk::{
    propagation::TraceContextPropagator,
    resource::Resource,
    runtime, // Use the runtime module
    trace::SdkTracerProvider,
};
use opentelemetry_semantic_conventions::resource::{SERVICE_NAME, SERVICE_VERSION};
use tracing::{Metadata, Subscriber};
use tracing_subscriber::layer::Filter;
use tracing_subscriber::prelude::__tracing_subscriber_SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::Layer;

pub struct LangfuseOptions<'a> {
    pub public_key: Option<&'a str>,
    pub secret_key: Option<&'a str>,
    pub host: Option<&'a str>,
}

/// Filter to exclude rmcp library internal spans that don't have proper parent context
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
        let name = meta.name();
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

    // 2. Build the ASYNC Processor
    // Now that we imported the correct struct, .builder() accepts 2 arguments!
    // This runs the export on the Tokio runtime, fixing the "no reactor" panic.
    let processor = BatchSpanProcessor::builder(exporter, runtime::Tokio).build();

    // 3. Provider Setup
    // Note: We use .with_span_processor(), NOT .with_batch_exporter()
    let provider = SdkTracerProvider::builder()
        .with_resource(resource)
        .with_span_processor(processor)
        .build();

    let tracer = provider.tracer("reagent-rs");
    global::set_tracer_provider(provider.clone());

    // 4. Layers (With the strict Allow-List filter we created)
    let otel_layer = tracing_opentelemetry::layer()
        .with_tracer(tracer)
        .with_filter(RmcpSpanFilter); // <--- KEEPS DEADLOCKS AWAY

    // 5. Console Filter (Clean output)
    let fmt_filter = EnvFilter::new("info");
    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_thread_ids(true)
        .with_target(true)
        .with_filter(fmt_filter);

    tracing_subscriber::registry()
        .with(fmt_layer)
        .with(otel_layer)
        .init();

    provider
}
