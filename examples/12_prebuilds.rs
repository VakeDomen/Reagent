use reagent_rs::{
    prebuilds::{StatefullPrebuild, StatelessPrebuild},
    AgentBuilder, McpServerType,
};
use std::error::Error;

const SCRAPER_AGENT_URL: &str = "http://localhost:8000/sse";
const MEMORY_URL: &str = "npx -y @<something/memory>";
const RAG_SERVICE: &str = "http://localhost:8001/mcp";

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    reagent_rs::observability::init_default_tracing();

    // StatefullPrebuild and StatefullPrebuild provide some
    // presets that you can use for simple things

    // diff between Stateless and Statefull is only that the
    // agent's history is reset on every invocation

    // simple agent whose flow is simply invoking the agent
    // and passing the message to the user. If the agent has access
    // to tools the message may contain tool calls but the user
    // has to invoke the actual tool themselves
    let _ = StatelessPrebuild::reply()
        .set_model("qwen3:0.6b")
        .build()
        .await?;

    let _ = StatefullPrebuild::reply()
        .set_model("qwen3:0.6b")
        .build()
        .await?;

    // simple agent whose flow is simply invoking the agent
    // and passing the message to the user. However even if
    // the agent has tools set, the request will not pass the
    // tool calls to the api. Returned is the message response of the
    // assistant
    let _ = StatelessPrebuild::reply_without_tools()
        .set_model("qwen3:0.6b")
        .build()
        .await?;

    let _ = StatefullPrebuild::reply_without_tools()
        .set_model("qwen3:0.6b")
        .build()
        .await?;

    // simple agent whose flow is simply invoking the agent
    // and passing the message to the user. If the agent has access
    // to tools the message may contain tool calls. In this case
    // however the agent will automatically invoke the tool calls.
    // if you use the StatefullPrebuild the history will not be reset
    // and the tool responses (Tool message) are pushed to the histroy
    let _ = StatelessPrebuild::call_tools()
        .set_model("qwen3:0.6b")
        .build()
        .await?;

    let _ = StatefullPrebuild::call_tools()
        .set_model("qwen3:0.6b")
        .build()
        .await?;

    // the default flow is a combination on call_tools and reply
    // first the agent will be invoked normally. if the agent decides
    // to call any tools, all tools will be invoked. The tools responses
    // will be added to the histroy of the agent and the agent will be invoked
    // again with reply_without_tools to respond with any summaries of the
    // tools
    let _ = AgentBuilder::default()
        .set_model("qwen3:0.6b")
        .build()
        .await?;

    // there is a plan-and-execute implementation present in the StatefullPrebuild
    // it is ment for quick testing. It is however recomended you implement own agents
    // and flows and not use prebuils as they may not be stable for your usecase and
    // currently does not yet support overriding system prompts of sub-agents
    let mut plan_and_execute_agent = StatefullPrebuild::plan_and_execute()
        .add_mcp_server(McpServerType::sse(SCRAPER_AGENT_URL))
        .add_mcp_server(McpServerType::stdio(MEMORY_URL))
        .add_mcp_server(McpServerType::streamable_http(RAG_SERVICE))
        .set_model("qwen3:30b")
        .build()
        .await?;

    let _ = plan_and_execute_agent
        .invoke_flow("Does this university offer any sholarships for PhD students?")
        .await;
    println!("histroy: {:#?}", plan_and_execute_agent.history);

    Ok(())
}
