use core::fmt;
use std::collections::HashMap;

use super::TemplateDataSource;


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
