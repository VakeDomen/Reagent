use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Registry};

pub fn init_default_tracing() {
    let filter = EnvFilter::builder()
        .with_default_directive(tracing::Level::WARN.into())
        .parse("reagent=debug,tool=info")
        .unwrap();

    let fmt_layer = fmt::layer()
        .with_target(false)
        .with_span_events(tracing_subscriber::fmt::format::FmtSpan::CLOSE);

    Registry::default().with(filter).with(fmt_layer).init();
}
