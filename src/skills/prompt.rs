use std::collections::HashMap;

use crate::{Template, SKILL_DISCOVERY_TEMPLATE};

use super::Skill;

impl Skill {
    pub async fn discovery_description(&self) -> String {
        let template = Template::simple(SKILL_DISCOVERY_TEMPLATE);

        let hm = HashMap::from([
            ("skill_name", &self.name),
            ("skill_description", &self.description),
        ]);

        template.compile(&hm).await
    }
}
