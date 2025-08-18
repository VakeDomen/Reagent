use std::{collections::HashMap, future::Future, pin::Pin};

/// A trait for providing dynamic values to templates.
/// (define once, call on every agent invocation)
///
/// Implementors of this trait act as a data source for template rendering,
/// supplying key-value pairs asynchronously.  
///
/// ```
/// use std::{collections::HashMap, future::Future, pin::Pin};
/// use reagent::templates::TemplateDataSource;
///
/// struct StaticDataSource;
///
/// impl TemplateDataSource for StaticDataSource {
///     fn get_values(&self) -> Pin<Box<dyn Future<Output = HashMap<String, String>> + Send>> {
///         Box::pin(async {
///             HashMap::from([
///                 ("name".to_string(), "Alice".to_string())
///             ])
///         })
///     }
///
///     fn clone_data_source(&self) -> Box<dyn TemplateDataSource> {
///         Box::new(StaticDataSource)
///     }
/// }
/// ```
pub trait TemplateDataSource: Send + Sync {
    /// Asynchronously fetches a map of key-value pairs to be injected into a template.
    ///
    /// # Returns
    /// A pinned, boxed future resolving to a `HashMap<String, String>` containing template values.
    fn get_values(&self) -> Pin<Box<dyn Future<Output = HashMap<String, String>> + Send>>;

    /// Creates a clone of the current data source.
    ///
    /// This is required because trait objects (`Box<dyn TemplateDataSource>`) don’t support `Clone`
    /// directly. Instead, implementors must define how they can be duplicated.
    fn clone_data_source(&self) -> Box<dyn TemplateDataSource>;
}
