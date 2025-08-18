use core::fmt;
use std::collections::HashMap;

use super::TemplateDataSource;

/// A lightweight text template with double-brace placeholders.
///
/// Placeholders use the form `{{key}}`. Values can come from:
/// 1. A dynamic [`TemplateDataSource`] provided at construction
/// 2. An explicit `HashMap<String, String>` passed to [`Template::compile`]
///
/// If both provide the same key, the explicit map passed to `compile` wins.
pub struct Template {
    content: String,
    data_source: Option<Box<dyn TemplateDataSource>>,
}

impl Template {
    /// Create a template with a dynamic data source.
    ///
    /// The `content` string may contain placeholders like `{{name}}`.
    /// The `data_source` is queried during [`compile`](Self::compile) to
    /// supply additional values.
    ///
    /// # Example
    /// ```
    /// use std::{collections::HashMap, future::Future, pin::Pin};
    /// use reagent::templates::{Template, TemplateDataSource};
    ///
    /// struct StaticDS;
    /// impl TemplateDataSource for StaticDS {
    ///     fn get_values(&self) -> Pin<Box<dyn Future<Output = HashMap<String, String>> + Send>> {
    ///         Box::pin(async {
    ///             let mut m = HashMap::new();
    ///             m.insert("name".into(), "Ada".into());
    ///             m
    ///         })
    ///     }
    ///     fn clone_data_source(&self) -> Box<dyn TemplateDataSource> { Box::new(StaticDS) }
    /// }
    ///
    /// async {
    ///     let t = Template::new("Hello, {{name}}!", StaticDS);
    ///     let out = t.compile(&HashMap::new()).await;
    ///     assert_eq!(out, "Hello, Ada!");
    /// };
    /// ```
    pub fn new<D: TemplateDataSource + 'static>(content: &str, data_source: D) -> Self {
        Self {
            content: content.to_string(),
            data_source: Some(Box::new(data_source)),
        }
    }

    /// Create a template without a data source.
    ///
    /// Useful for static templates where all values will be supplied
    /// at call time via [`compile`](Self::compile).
    pub fn simple<T>(content: T) -> Self where T: Into<String> {
        Self { content: content.into(), data_source: None }
    }

    /// Render the template by replacing placeholders with values.
    ///
    /// The lookup order is:
    /// 1. Values from the optional data source
    /// 2. Values from the provided `data` map, which override duplicates
    ///
    /// Any placeholders without a matching key remain unchanged.
    pub async fn compile(&self, data: &HashMap<String, String>) -> String {
        let mut filled_content = self.content.clone();

        if let Some(source) = &self.data_source {
            let generated_data = source.get_values().await;
            for (key, value) in generated_data {
                let placeholder = format!("{{{{{key}}}}}");
                filled_content = filled_content.replace(&placeholder, &value);
            }
        }

        for (key, value) in data {
            let placeholder = format!("{{{{{key}}}}}");
            filled_content = filled_content.replace(&placeholder, value);
        }

        filled_content
    }
}

impl Clone for Template {
    fn clone(&self) -> Self {
        Self {
            content: self.content.clone(),
            data_source: match &self.data_source {
                None => None,
                Some(data_source) => Some(data_source.clone_data_source())
            },
        }
    }
}

impl fmt::Debug for Template {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Template")
            .field("content", &self.content)
            .field("data_source", &self.data_source.as_ref().map(|_| "Some(Box<dyn TemplateDataSource>)").unwrap_or("None"))
            .finish()
    }
}
