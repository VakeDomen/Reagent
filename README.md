# Reagent

Reagent is a Rust library for building and running AI agents that interact with LLMs. It abstracts away provider-specific details (currently supports [Ollama](https://ollama.com) and [OpenRouter](https://openrouter.ai)), provides a consistent API for prompting, structured outputs, and tool use, and allows you to define fully custom invocation flows.

You can add the library to your project by pulling from crates:

```bash
cargo add reagent-rs
```

or directly from github:

```toml
[dependencies]
reagent = { git = "https://github.com/VakeDomen/Reagent" }
```
---

## Notes

* Reagent is experimental and provider support may change.
* Not all provider features are unified;


---

## Features

* **Multiple providers**: Ollama (default) and OpenRouter (experimental)
* **Structured output** via JSON Schema (manual or via `schemars`)
* **Tooling**:

  * Define tools with input schemas
  * Register async executors for tool calls
  * Integrate MCP (Model Context Protocol) servers as tools
* **Flows**:

  * Default flows for common patterns
  * Custom flows and closures
  * Prebuilt flows for quick prototyping
* **Prompt templates** with runtime or dynamic data sources
* **Notifications**: subscribe to agent events like token streaming, tool calls, errors, etc.

---

## Quick Start

```rust
use std::error::Error;
use reagent_rs::AgentBuilder;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {

    let mut agent = AgentBuilder::default()
        .set_model("qwen3:0.6b")
        .set_system_prompt("You are a helpful assistant.")
        .build()
        .await?;

    let resp = agent.invoke_flow("Hello!").await?;
    println!("Agent response: {}", resp.content.unwrap_or_default());

    Ok(())
}
```

---

## Building Agents

The `AgentBuilder` uses a builder pattern. Only `model` is required; everything else has defaults.

```rust
let agent = AgentBuilder::default()
    .set_model("qwen3:0.6b")
    .set_system_prompt("You are a helpful assistant.")
    .set_temperature(0.7)
    .set_num_ctx(2048)
    .build()
    .await?;
```

### Providers

By default, Reagent assumes an Ollama instance running locally.

```rust
let agent = AgentBuilder::default()
    .set_model("qwen3:0.6b")
    .set_provider(Provider::Ollama)
    .set_base_url("http://localhost:11434")
    .build()
    .await?;
```

To use OpenRouter:

```rust
let agent = AgentBuilder::default()
    .set_model("qwen3:0.6b")
    .set_provider(Provider::OpenRouter)
    .set_api_key("YOUR_KEY")
    .build()
    .await?;
```

Note: some providers require provider-specific response format settings.

---

## Structured Output

You can ask the model to return JSON that matches a schema.

Manual schema:

```rust
let agent = AgentBuilder::default()
    .set_model("qwen3:0.6b")
    .set_response_format(r#"{
        "type":"object",
        "properties":{
            "windy":{"type":"boolean"},
            "temperature":{"type":"integer"},
            "description":{"type":"string"}
        },
        "required":["windy","temperature","description"]
    }"#)
    .build()
    .await?;
```

From struct via `schemars`:

```rust
#[derive(JsonSchema)]
struct Weather { 
    windy: bool, 
    temperature: i32, 
    description: String 
}

let agent = AgentBuilder::default()
    .set_model("qwen3:0.6b")
    .set_response_format(serde_json::to_string_pretty(&schema_for!(Weather))?)
    .build()
    .await?;
```

To get parsed output directly:

```rust
let resp: Weather = agent.invoke_flow_structured_output("What's the weather?").await?;
```

---

## Tools

Tools let the model call custom functions. Define an executor closure, wrap it in a `ToolBuilder`, and register it with the agent.

```rust
async fn get_weather(args: Value) -> Result<String, ToolExecutionError> {
    // do your thing
    Ok(r#"{"windy":false,"temperature":18}"#.into())
};

let tool = ToolBuilder::new()
    .function_name("get_weather")
    .add_required_property("location", "string", "City name")
    .executor_fn(get_weather)
    .build()?;

let agent = AgentBuilder::default()
    .set_model("qwen3:0.6b")
    .add_tool(tool)
    .add_mcp_server(McpServerType::sse("http://localhost:8000/sse"))
    .add_mcp_server(McpServerType::stdio("npx -y @<something/memory>"))
    .add_mcp_server(McpServerType::streamable_http("http://localhost:8001/mcp"))
    .build()
    .await?;
```

---

## Flows

Flows control how the agent is invoked.

* **Default flow**: prompt -> LLM -> (maybe tool call -> LLM) -> result
* **Prebuilt flows**: e.g., `reply`, `reply_without_tools`, `call_tools`, `plan_and_execute`
* **Custom flow functions**:

```rust
async fn my_custom_flow(agent: &mut Agent, prompt: String) -> Result<Message, AgentError> {
    // custom logic
    Ok(Message::assistant("Hello"))
}

let agent = AgentBuilder::default()
    .set_model("qwen3:0.6b")
    .set_flow(flow!(my_flow))
    .build()
    .await?;
```

---

## Templates

Define prompts with placeholders:

```rust
let template = Template::simple("Hello {{name}}!");

let agent = AgentBuilder::default()
    .set_model("qwen3:0.6b")
    .set_template(template)
    .build()
    .await?;

let prompt_data = HashMap::from([
    ("name", "Peter"),
]);

let resp = agent.invoke_flow_with_template(prompt_data).await?;
```

Pass a `HashMap` of values to `invoke_flow_with_template`.

You can also provide a `TemplateDataSource` that injects dynamic values at invocation time.

---

## Notifications & Streaming

You can receive events from the agent using `build_with_notification`:

```rust
let (agent, mut rx) = AgentBuilder::default()
    .set_model("qwen3:0.6b")
    .set_stream(true)
    .build_with_notification()
    .await?;
```


---

## Prebuilds

For quick experiments, `StatelessPrebuild` and `StatefullPrebuild` offer presets some simple flow patterns. Stateful versions keep conversation history; stateless ones reset each call.

Examples:

```rust
let agent = StatelessPrebuild::reply()
    .set_model("qwen3:0.6b")
    .build()
    .await?;
let agent = StatefullPrebuild::call_tools()
    .set_model("qwen3:0.6b")
    .build()
    .await?;
```

---

## License

MIT
