
use std::error::Error;
use reagent::{init_default_tracing, AgentBuilder, McpServerType};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    init_default_tracing();

    let _agent = AgentBuilder::default()
        .set_model("qwen3:0.6b")
        .set_system_prompt("You are a helpful asistant")
        .add_mcp_server(McpServerType::Sse("http://localhost:8001/sse".into()))
        .add_mcp_server(McpServerType::sse("http://localhost:8001/sse"))
        .add_mcp_server(McpServerType::stdio("npx -y @modelcontextprotocol/server-memory"))
        .add_mcp_server(McpServerType::streamable_http("http://localhost:3000/connect"))
        .build()
        .await?;

    Ok(())
}
