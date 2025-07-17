use std::collections::HashMap;

/// A trait defining an asynchronous data source that provides key-value pairs.
pub trait TemplateDataSource {
    /// Asynchronously retrieves a HashMap of key-value pairs to be replaced in the template
    fn get_values(&self) -> impl std::future::Future<Output = HashMap<String, String>> + Send;
}

/// A struct representing a template that can be filled with dynamic data.
///
/// # Type Parameters
///
/// * `T`: A type that implements the `DataSource` trait.
#[derive(Clone)]
pub struct Template<T> where T: TemplateDataSource {
    content: String,
    data_source: T,
}

impl<T: TemplateDataSource> Template<T> {
    /// Creates a new `Template` with the given content and data source.
    ///
    /// # Arguments
    ///
    /// * `content` - A string slice that holds the template content.
    /// * `data_source` - A data source that implements the `DataSource` trait.
    ///
    /// # Returns
    ///
    /// A new instance of `Template`.
    pub fn new(content: &str, data_source: T) -> Self {
        Self {
            content: content.to_string(),
            data_source,
        }
    }

    /// Asynchronously compiles the template by replacing placeholders with values from the provided data and data source.
    ///
    /// # Arguments
    ///
    /// * `data` - A HashMap containing key-value pairs to replace placeholders in the template.
    ///
    /// # Returns
    ///
    /// A `String` with the compiled template content.
    pub async fn compile(&self, data: &HashMap<String, String>) -> String {
        let mut filled_content = self.content.clone();
        let generated_data = self.data_source.get_values().await;

        for (key, value) in generated_data {
            let placeholder = format!("{{{{{}}}}}", key);
            filled_content = filled_content.replace(&placeholder, &value);
        }

        for (key, value) in data {
            let placeholder = format!("{{{{{}}}}}", key);
            filled_content = filled_content.replace(&placeholder, value);
        }

        filled_content
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    pub struct MockDataSource {
        data: HashMap<String, String>,
    }

    impl TemplateDataSource for MockDataSource {
        async fn get_values(&self) -> HashMap<String, String> {
            self.data.clone()
        }
    }

    #[tokio::test]
    async fn test_template_compile() {
        let mock_data_source = MockDataSource {
            data: {
                let mut data = HashMap::new();
                data.insert("datetime".to_string(), "2023-10-01T12:00:00".to_string());
                data
            },
        };

        let template = Template::new(
            "Current datetime is: {{datetime}}. User input: {{user_input}}.",
            mock_data_source,
        );

        let mut user_data = HashMap::new();
        user_data.insert("user_input".to_string(), "Hello, world!".to_string());

        let compiled_template = template.compile(&user_data).await;
        assert_eq!(compiled_template, "Current datetime is: 2023-10-01T12:00:00. User input: Hello, world!.");
    }

    #[tokio::test]
    async fn test_template_compile_without_user_data() {
        let mock_data_source = MockDataSource {
            data: {
                let mut data = HashMap::new();
                data.insert("datetime".to_string(), "2023-10-01T12:00:00".to_string());
                data
            },
        };

        let template = Template::new(
            "Current datetime is: {{datetime}}.",
            mock_data_source,
        );

        let user_data = HashMap::new();
        let compiled_template = template.compile(&user_data).await;
        assert_eq!(compiled_template, "Current datetime is: 2023-10-01T12:00:00.");
    }
}