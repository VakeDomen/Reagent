
use std::error::Error;
use reagent::{init_default_tracing, AgentBuilder, McpServerType};


const SCRAPER_AGENT_URL: &str = "http://localhost:8000/sse"; 
const STAFF_AGENT_URL: &str = "http://localhost:8001/sse";
const MEMORY_URL: &str = "http://localhost:8002/sse";
const PROGRAMME_AGENT_URL: &str = "http://localhost:8003/sse";
const RAG_SERVICE: &str = "http://localhost:8005/sse"; 


#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    init_default_tracing();
    

    let mut plan_and_execute_agent = AgentBuilder::plan_and_execute()
        .add_mcp_server(McpServerType::Sse(STAFF_AGENT_URL.into()))
        .add_mcp_server(McpServerType::Sse(PROGRAMME_AGENT_URL.into()))
        .add_mcp_server(McpServerType::Sse(SCRAPER_AGENT_URL.into()))
        .add_mcp_server(McpServerType::Sse(MEMORY_URL.into()))
        .add_mcp_server(McpServerType::Sse(RAG_SERVICE.into()))
        .set_model("qwen3:30b")
        .build()
        .await?;


    let _ = plan_and_execute_agent.invoke_flow("Does this university offer any sholarships for PhD students?").await;
    println!("histroy: {:#?}", plan_and_execute_agent.history);

    Ok(())
}

