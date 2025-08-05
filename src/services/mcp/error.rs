use std::fmt;


// Custom error for MCP integration
#[derive(Debug)]
pub enum McpIntegrationError {
    Sdk(rmcp::Error), // Assuming mcp_sdk::Error is a concrete type
    Connection(String),
    Discovery(String),
    ToolConversion(String),
    InvalidSchema(String),
}

// Implement Display for McpIntegrationError
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

// Implement Error for McpIntegrationError
impl std::error::Error for McpIntegrationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            McpIntegrationError::Sdk(e) => Some(e), // `e` must be 'static or live long enough
            // Other variants currently don't wrap another error directly in a way
            // that `source()` can return, unless their String content is from an error.
            // If the String fields store error messages from other errors, you might not
            // be able to return the original error object here without boxing or changing structure.
            McpIntegrationError::Connection(_) => None,
            McpIntegrationError::Discovery(_) => None,
            McpIntegrationError::ToolConversion(_) => None,
            McpIntegrationError::InvalidSchema(_) => None,
        }
    }
}

// Implement From<mcp_sdk::Error> for McpIntegrationError for easy conversion (like `?` operator)
// This replaces the `#[from]` attribute that `thiserror` provides.
impl From<rmcp::Error> for McpIntegrationError {
    fn from(err: rmcp::Error) -> Self {
        McpIntegrationError::Sdk(err)
    }
}