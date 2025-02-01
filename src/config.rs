use serde::Deserialize;
use anyhow::Result;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub server: ServerConfig,
    pub model: ModelConfig,
}

#[derive(Debug, Deserialize)]
pub struct ServerConfig {
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
}

#[derive(Debug, Deserialize)]
pub struct ModelConfig {
    pub model_id: String,
    #[serde(default = "default_revision")]
    pub revision: String,
}

fn default_host() -> String {
    "127.0.0.1".to_string()
}

fn default_port() -> u16 {
    3000
}

fn default_revision() -> String {
    "main".to_string()
}

impl Config {
    pub fn load() -> Result<Self> {
        let config = config::Config::builder()
            .add_source(config::File::with_name("config").required(false))
            .add_source(config::Environment::with_prefix("INFRS"))
            .build()?;

        Ok(config.try_deserialize()?)
    }

    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let config = config::Config::builder()
            .add_source(config::File::from(path.as_ref()))
            .add_source(config::Environment::with_prefix("INFRS"))
            .build()?;

        Ok(config.try_deserialize()?)
    }
}
