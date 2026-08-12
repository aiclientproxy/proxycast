use super::CanonicalModel;
use anyhow::{Context, Result};
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::path::Path;

/// Cached bundled canonical model registry
static BUNDLED_REGISTRY: Lazy<Result<CanonicalModelRegistry>> = Lazy::new(|| {
    const CANONICAL_MODELS_JSON: &str = include_str!("data/canonical_models.json");

    let models: Vec<CanonicalModel> = serde_json::from_str(CANONICAL_MODELS_JSON)
        .context("Failed to parse bundled canonical models JSON")?;

    let mut registry = CanonicalModelRegistry::new();
    for model in models {
        registry.register(model)?;
    }

    Ok(registry)
});

#[derive(Debug, Clone)]
pub struct CanonicalModelRegistry {
    models: HashMap<String, CanonicalModel>,
}

impl CanonicalModelRegistry {
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
        }
    }

    pub fn bundled() -> Result<&'static Self> {
        BUNDLED_REGISTRY
            .as_ref()
            .map_err(|e| anyhow::anyhow!("{}", e))
    }

    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref())
            .context("Failed to read canonical models file")?;

        let models: Vec<CanonicalModel> =
            serde_json::from_str(&content).context("Failed to parse canonical models JSON")?;

        let mut registry = Self::new();
        for model in models {
            registry.register(model)?;
        }

        Ok(registry)
    }

    pub fn to_file(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut models: Vec<&CanonicalModel> = self.models.values().collect();
        models.sort_by(|a, b| a.id.cmp(&b.id));

        let json = serde_json::to_string_pretty(&models)
            .context("Failed to serialize canonical models")?;

        std::fs::write(path.as_ref(), json).context("Failed to write canonical models file")?;

        Ok(())
    }

    pub fn register(&mut self, model: CanonicalModel) -> Result<()> {
        if model.id.trim().is_empty() {
            anyhow::bail!("Canonical model id must not be empty");
        }
        if self.models.contains_key(&model.id) {
            anyhow::bail!("Duplicate canonical model id: {}", model.id);
        }
        self.models.insert(model.id.clone(), model);
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<&CanonicalModel> {
        self.models.get(name)
    }

    pub fn all_models(&self) -> Vec<&CanonicalModel> {
        let mut models: Vec<_> = self.models.values().collect();
        models.sort_by(|left, right| left.id.cmp(&right.id));
        models
    }

    pub fn count(&self) -> usize {
        self.models.len()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.models.contains_key(name)
    }
}

impl Default for CanonicalModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::super::Pricing;
    use super::*;

    fn model(id: &str) -> CanonicalModel {
        CanonicalModel {
            id: id.to_string(),
            name: id.to_string(),
            context_length: 1,
            max_completion_tokens: None,
            task_families: Vec::new(),
            input_modalities: vec!["text".to_string()],
            output_modalities: vec!["text".to_string()],
            runtime_features: Vec::new(),
            tool_mode: None,
            supports_tools: false,
            supports_reasoning: false,
            supports_prompt_cache: false,
            pricing: Pricing {
                prompt: None,
                completion: None,
                request: None,
                image: None,
            },
        }
    }

    #[test]
    fn all_models_returns_stable_id_order() {
        let mut registry = CanonicalModelRegistry::new();
        registry.register(model("provider/zeta")).unwrap();
        registry.register(model("provider/alpha")).unwrap();
        registry.register(model("provider/mid")).unwrap();

        let ids: Vec<_> = registry
            .all_models()
            .into_iter()
            .map(|model| model.id.as_str())
            .collect();

        assert_eq!(ids, vec!["provider/alpha", "provider/mid", "provider/zeta"]);
    }

    #[test]
    fn register_rejects_ambiguous_model_identity() {
        let mut registry = CanonicalModelRegistry::new();
        registry.register(model("provider/alpha")).unwrap();

        let duplicate = registry.register(model("provider/alpha")).unwrap_err();
        let blank = registry.register(model("  ")).unwrap_err();

        assert_eq!(
            duplicate.to_string(),
            "Duplicate canonical model id: provider/alpha"
        );
        assert_eq!(blank.to_string(), "Canonical model id must not be empty");
        assert_eq!(registry.count(), 1);
    }
}
