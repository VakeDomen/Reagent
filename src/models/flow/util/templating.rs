use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

pub trait TemplateDataSource: Send {
    fn get_values(&self) -> Pin<Box<dyn Future<Output = HashMap<String, String>> + Send>>;
    fn clone_data_source(&self) -> Box<dyn TemplateDataSource>;
}

pub struct Template {
    content: String,
    data_source: Option<Box<dyn TemplateDataSource>>,
}

impl Template {
    pub fn new<D: TemplateDataSource + 'static>(content: &str, data_source: D) -> Self {
        Self {
            content: content.to_string(),
            data_source: Some(Box::new(data_source)),
        }
    }

    pub fn simple<T>(content: T) -> Self where T: Into<String> {
        Self { content: content.into(), data_source: None }
    }

    pub async fn compile(&self, data: &HashMap<String, String>) -> String {
        let mut filled_content = self.content.clone();

        if let Some(source) = &self.data_source {
            let generated_data = source.get_values().await;
            for (key, value) in generated_data {
                let placeholder = format!("{{{{{}}}}}", key);
                filled_content = filled_content.replace(&placeholder, &value);
            }
        }

        for (key, value) in data {
            let placeholder = format!("{{{{{}}}}}", key);
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[derive(Clone)]
    pub struct MockDataSource {
        data: HashMap<String, String>,
    }

    impl MockDataSource {
        pub fn new(data: HashMap<String, String>) -> Self {
            Self { data }
        }
    }

    impl TemplateDataSource for MockDataSource {
        fn get_values(&self) -> Pin<Box<dyn Future<Output = HashMap<String, String>> + Send>> {
            let data = self.data.clone();
            Box::pin(async move { data })
        }

        fn clone_data_source(&self) -> Box<dyn TemplateDataSource> {
            Box::new(self.clone())
        }
    }

    #[tokio::test]
    async fn test_template_compile() {
        let mock_data_source = MockDataSource::new({
            let mut data = HashMap::new();
            data.insert("datetime".to_string(), "2023-10-01T12:00:00".to_string());
            data
        });

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
        let mock_data_source = MockDataSource::new({
            let mut data = HashMap::new();
            data.insert("datetime".to_string(), "2023-10-01T12:00:00".to_string());
            data
        });

        let template = Template::new(
            "Current datetime is: {{datetime}}.",
            mock_data_source,
        );

        let user_data = HashMap::new();
        let compiled_template = template.compile(&user_data).await;
        assert_eq!(compiled_template, "Current datetime is: 2023-10-01T12:00:00.");
    }
}
