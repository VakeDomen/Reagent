use rmcp::schemars::{gen::SchemaSettings, schema::RootSchema, JsonSchema, SchemaGenerator};
use serde_json::Value;

#[derive(Clone, Debug)]
pub struct SchemaSpec {
    pub schema: Value,        // pure JSON Schema root
    pub name: Option<String>, // used by providers that want a name
    pub strict: Option<bool>, // opt-in, only applied where supported
}

impl SchemaSpec {
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
    pub fn strict(mut self, strict: bool) -> Self {
        self.strict = Some(strict);
        self
    }

    pub fn from_value(schema: serde_json::Value) -> Self {
        Self {
            schema,
            name: None,
            strict: None,
        }
    }
    pub fn from_str(s: &str) -> Result<Self, serde_json::Error> {
        let v: serde_json::Value = serde_json::from_str(s.trim())?;
        Ok(Self::from_value(v))
    }
    pub fn from_type<T: JsonSchema>() -> Self {
        let settings = SchemaSettings::draft07().with(|s| {
            s.inline_subschemas = true;
            s.meta_schema = None;
        });
        let gen = SchemaGenerator::new(settings);
        let root: RootSchema = gen.into_root_schema_for::<T>();
        let mut schema = serde_json::to_value(&root.schema).expect("schema serialize");
        if let Some(obj) = schema.as_object_mut() {
            obj.remove("$schema");
            obj.remove("definitions");
        }
        Self::from_value(schema)
    }
}

#[derive(Clone, Debug, Default)]
pub struct ResponseFormatConfig {
    spec: Option<SchemaSpec>,
    raw: Option<String>,
    name: Option<String>,
    strict: Option<bool>,
}

impl ResponseFormatConfig {
    pub fn set_raw(&mut self, schema_json: impl Into<String>) {
        self.raw = Some(schema_json.into());
    }

    pub fn set_value(&mut self, schema: serde_json::Value) {
        self.spec = Some(SchemaSpec::from_value(schema));
    }

    pub fn set_type<T: JsonSchema>(&mut self) {
        self.spec = Some(SchemaSpec::from_type::<T>());
    }

    pub fn set_spec(&mut self, spec: SchemaSpec) {
        self.spec = Some(spec);
    }

    pub fn set_name(&mut self, name: impl Into<String>) {
        self.name = Some(name.into());
    }

    pub fn set_strict(&mut self, strict: bool) {
        self.strict = Some(strict);
    }

    pub fn resolve(self) -> Result<Option<SchemaSpec>, String> {
        if self.spec.is_some() && self.raw.is_some() {
            return Err(
                "Both set_structured_output_* and set_response_format_str were called. Use only one source."
                    .to_string(),
            );
        }

        let Some(mut spec) = self.spec else {
            let Some(raw) = self.raw else {
                return Ok(None);
            };

            let schema = serde_json::from_str(raw.trim())
                .map_err(|e| format!("Failed to parse JSON schema: {e}"))?;
            return Ok(Some(SchemaSpec {
                schema,
                name: self.name,
                strict: self.strict,
            }));
        };

        spec.name = self.name.or(spec.name);
        spec.strict = self.strict.or(spec.strict);
        Ok(Some(spec))
    }
}

pub trait StructuredOuputFormat {
    fn format(spec: &SchemaSpec) -> serde_json::Value;
}
