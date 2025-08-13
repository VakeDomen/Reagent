use std::{collections::HashMap, future::Future, pin::Pin};

pub trait TemplateDataSource: Send + Sync {
    fn get_values(&self) -> Pin<Box<dyn Future<Output = HashMap<String, String>> + Send>>;
    fn clone_data_source(&self) -> Box<dyn TemplateDataSource>;
}
