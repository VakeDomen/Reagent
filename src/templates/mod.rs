mod data_source;
mod template;

pub use self::{
    data_source::TemplateDataSource,
    template::Template
};


#[cfg(test)]
mod tests {
    use super::*;
    use std::{collections::HashMap, future::Future, pin::Pin};

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
