# Reagent

Reagent is a Rust library for building and running AI agents that interact with LLMs. It abstracts away provider-specific details for [Ollama](https://ollama.com), [OpenRouter](https://openrouter.ai), and OpenAI-compatible endpoints; provides a consistent API for prompting, structured outputs, and tool use; and allows you to define custom agent flows.

You can add the library to your project by pulling from crates:

```bash
cargo add reagent-rs
```

or directly from github:

```toml
[dependencies]
reagent-rs = { git = "https://github.com/VakeDomen/Reagent" }
```
---

## Notes

* Reagent is experimental and provider support may change.
* Not all provider features are unified;


---

## Features

* **Multiple providers**: Ollama (default), OpenRouter, and OpenAI-compatible endpoints
* **Reusable stateless models** with typed chat and embedding invocations
* **Images** via `Message::with_image` and multi-modal model inputs
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

    let resp = agent.invoke("Hello!").await?;
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

For a typed response, define the shape with `schemars` and select it on the
builder. The agent's `invoke` method then returns `Message<T>`.

```rust
use reagent_rs::{AgentBuilder, JsonSchema, Message};
use serde::Deserialize;

#[derive(Debug, Deserialize, JsonSchema)]
struct Weather {
    windy: bool,
    temperature: i32,
    description: String
}

let mut agent = AgentBuilder::default()
    .set_model("qwen3:0.6b")
    .structured_output::<Weather>()
    .build()
    .await?;

let response: Message<Weather> = agent.invoke("What's the weather?").await?;
```

When you provide a schema directly without an infered type (same on `Model`
builder). The response content is then `serde_json::Value`:

```rust
use reagent_rs::{ChatResponse, Invocation, Message, Value};

let response: ChatResponse<Value> = Invocation::chat()
    .model("qwen3:0.6b")
    .message(Message::user("What's the weather?"))
    .response_format_str(r#"{
        "type":"object",
        "properties":{"temperature":{"type":"integer"}}
    }"#)
    .invoke()
    .await?;
```

---

## Models and Invocations

`Invocation` is the fully configured, one-off API call. `Model` is the easier,
reusable API. it owns defaults and optional configured history, but never keeps
the prompt or response from an individual call. Models are typed by capability:
use `Model::llm(...)` for chat and `Model::embedding(...)` for embeddings.
The same model can be called repeatedly without accumulating any state.
Agents can reuse that model while keeping independent histories:

```rust
use reagent_rs::Model;

let model = Model::llm("qwen3:0.6b").build()?;
let chat = model.invoke("Hello").await?;
```

Models can use the same typed structured-output path as agents:

```rust
use reagent_rs::{ChatResponse, JsonSchema, Model};
use serde::Deserialize;

#[derive(Deserialize, JsonSchema)]
struct Entities {
    people: Vec<String>,
}

let extractor = Model::llm("qwen3:0.6b")
    .structured_output::<Entities>()
    .build()?;
let response: ChatResponse<Entities> = extractor.invoke("Ada visited London.").await?;
```

Models can also render a template for each invocation. A templated model accepts
template data instead of a prompt string, while still retaining neither the
rendered prompt nor the response:

```rust
use std::collections::HashMap;
use reagent_rs::Model;

let extractor = Model::llm("qwen3:0.6b")
    .set_template_simple("Extract entities from: {{text}}")
    .build()?;

let response = extractor
    .invoke(HashMap::from([("text", "Ada visited London.")]))
    .await?;
```

Use `set_template`, `set_template_simple`, `set_template_with_source`,
`set_template_from_file`, or `set_template_from_file_with_source` to configure
the template.

```rust
use reagent_rs::AgentBuilder;

let mut agent = AgentBuilder::default()
    .with_model(model.clone())
    .set_system_prompt("You are a helpful assistant.")
    .build()
    .await?;
```

For embeddings:

```rust
use reagent_rs::Model;

let model = Model::embedding("bge-m3").build()?;
let resp = model.invoke(["first text", "second text"]).await?;

println!("{} embeddings returned", resp.embeddings.len());
```

For a direct, one-off call, configure and invoke an `Invocation` directly:

```rust
use reagent_rs::{Invocation, Message, Provider};

let chat = Invocation::chat()
    .model("gpt-4o-mini")
    .provider(Provider::OpenAi)
    .base_url("https://api.example.com/v1")
    .message(Message::user("Hello"))
    .invoke()
    .await?;
```

Images are attached to messages directly:

```rust
use reagent_rs::{Message, Model};

let model = Model::llm("llava").build()?;
let resp = model
    .invoke(Message::user("Describe the image").with_image("BASE64_DATA"))
    .await?;
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
* **Custom flow functions**:

```rust
async fn my_custom_flow(agent: &mut Agent, prompt: String) -> Result<Message, AgentError> {
    // custom logic
    Ok(Message::assistant("Hello"))
}

let agent = AgentBuilder::default()
    .set_model("qwen3:0.6b")
    .set_flow(flow!(my_custom_flow))
    .build()
    .await?;
```

---

## Templates

Define prompts with placeholders:

```rust
let template = Template::simple("Hello {{name}}!");

let mut agent = AgentBuilder::default()
    .set_model("qwen3:0.6b")
    .set_template(template)
    .build()
    .await?;

let prompt_data = HashMap::from([
    ("name", "Peter"),
]);

let resp = agent.invoke(prompt_data).await?;
```

Pass a `HashMap` of values to `invoke`. Calling `set_template` changes the
agent's invocation input from a prompt string to template data.

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

## License

MIT
