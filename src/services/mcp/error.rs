use std::fmt;


#[derive(Debug)]
pub enum McpIntegrationError {
    Sdk(rmcp::Error),
    Connection(String),
    Discovery(String),
    ToolConversion(String),
    InvalidSchema(String),
}

impl fmt::Display for McpIntegrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            McpIntegrationError::Sdk(e) => write!(f, "MCP SDK error: {e}"),
            McpIntegrationError::Connection(s) => {
                write!(f, "Failed to connect to MCP server: {s}")
            }
            McpIntegrationError::Discovery(s) => {
                write!(f, "Failed to discover MCP actions: {s}")
            }
            McpIntegrationError::ToolConversion(s) => {
                write!(f, "Failed to convert MCP action to agent tool: {s}")
            }
            McpIntegrationError::InvalidSchema(s) => {
                write!(f, "MCP action input schema is missing or not an object: {s}")
            }
        }
    }
}

impl std::error::Error for McpIntegrationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            McpIntegrationError::Sdk(e) => Some(e),
            McpIntegrationError::Connection(_) => None,
            McpIntegrationError::Discovery(_) => None,
            McpIntegrationError::ToolConversion(_) => None,
            McpIntegrationError::InvalidSchema(_) => None,
        }
    }
}

impl From<rmcp::Error> for McpIntegrationError {
    fn from(err: rmcp::Error) -> Self {
        McpIntegrationError::Sdk(err)
    }
}