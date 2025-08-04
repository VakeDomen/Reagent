
use std::error::Error;
use reagent::{init_default_tracing, AgentBuilder, McpServerType};

const SCRAPER_AGENT_URL: &str = "http://localhost:8000/sse"; 
const MEMORY_URL: &str = "npx -y @<something/memory>";
const RAG_SERVICE: &str = "http://localhost:8001/mcp"; 

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    init_default_tracing();

    let _agent = AgentBuilder::default()
        .set_model("qwen3:0.6b")
        .set_system_prompt("You are a helpful asistant")
        // you can add external mcp servers to the agent
        // on .build() the agent will connect to the server
        // retrieve the tools and add the to the list of 
        // avalible tools to use
        .add_mcp_server(McpServerType::sse(SCRAPER_AGENT_URL))
        .add_mcp_server(McpServerType::stdio(MEMORY_URL))
        .add_mcp_server(McpServerType::streamable_http(RAG_SERVICE))
        .build()
        .await?;

    Ok(())
}
