# Ragent - Simple rust Agents

**Rust Ollama Agent Library**

A zero-boilerplate Rust library for building simple AI agents on top of Ollama. It provides a flexible builder-pattern API for defining system prompts, JSON-schema response formats, and both static and dynamic tools (MCP and custom tool-calling). Designed for simplicity and extensibility.



## 📖 Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Getting Started](#getting-started)
5. [Examples](#examples)
6. [API Reference](#api-reference)
7. [Contributing](#contributing)
8. [License](#license)

---

## Introduction

**Ragent** is a lightweight Rust library that abstracts away the boilerplate of interacting with Ollama language models. With a flexible `AgentBuilder` and `ToolBuilder`, you can quickly assemble:

* A core agent powered by any Ollama model
* Custom tools with arbitrary async executors
* Dynamic tool discovery via MCP (Model Context Protocol) servers
* Structured responses enforced by JSON Schema

Whether you need a single-purpose assistant or a complex multi-tool pipeline, **Ragent** makes it straightforward.

---

## Features

* **Builder Pattern API**
  Configure every aspect of your agent with ergonomic, chainable methods.

* **Custom Tools**
  Define function signatures and attach async Rust executors via `ToolBuilder`.

* **MCP Tool Integration**
  Automatically discover and load tools exposed by local or remote MCP servers.

* **JSON-Schema Response Formats**
  Enforce structured output from the LLM for reliable downstream parsing.

* **Default-Friendly**
  Sensible defaults for endpoint, port, and system prompt reduce initial setup.

* **Async & Tokio-Ready**
  Non-blocking, thread-safe design with `Arc<Mutex<Agent>>` support for shared tools.

> *Streaming support is on our roadmap for v0.2.0*

---

## Installation

Add **Ragent** to your `Cargo.toml`:

```toml
[dependencies]
Ragent = "0.1"
```

Or via the CLI:

```bash
cargo add Ragent
```

Ensure you have a compatible Rust toolchain (Rust 1.65+).

---

## Getting Started

1. **Define Sub-Agents as Tools**
   You can encapsulate mini-agents—each with its own model, prompt, and schema—as reusable tools.

2. **Build the Main Agent**
   Compose your main agent with system prompt, attached tools, and optional MCP servers.

3. **Invoke and React**
   Use `agent.invoke("...")` to ask questions; tool calls are handled transparently.

### Basic Example

```rust
use Ragent::{AgentBuilder, ToolBuilder, AsyncToolFn, McpServerType};
use std::sync::Arc;
use tokio::sync::Mutex;
use serde_json::Value;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. Create a mini "weather" agent as a tool
    let weather_agent = AgentBuilder::default()
        .set_model("qwen3:30b")
        .set_system_prompt("/no_think \nAnswer in JSON: { wind, temp, desc }")
        .set_response_format(r#"
            {
              "type":"object",
              "properties":{ "wind":{"type":"integer"}, "temp":{"type":"integer"}, "desc":{"type":"string"} },
              "required":["wind","temp","desc"]
            }
        "#)
        .build()
        .await?;

    let weather_ref = Arc::new(Mutex::new(weather_agent));
    let weather_exec: AsyncToolFn = {
        let weather_ref = Arc::clone(&weather_ref);
        Arc::new(move |args: Value| {
            let weather_ref = Arc::clone(&weather_ref);
            Box::pin(async move {
                let mut agent = weather_ref.lock().await;
                let loc = args.get("location").and_then(|v| v.as_str())
                    .ok_or("Missing 'location' arg".to_string())?;
                let prompt = format!("/no_think What is the weather in {}?", loc);
                let res = agent.invoke(prompt).await.map_err(|e| e.to_string())?;
                Ok(res.content.unwrap_or_default())
            })
        })
    };

    let weather_tool = ToolBuilder::new()
        .function_name("get_current_weather")
        .function_description("Get a made-up weather forecast for a city")
        .add_property("location", "string", "City name")
        .add_required_property("location")
        .executor(weather_exec)
        .build()?;

    // 2. Compose the primary agent with MCP and custom tools
    let mut agent = AgentBuilder::default()
        .set_model("qwen3:30b")
        .set_system_prompt("/no_think \nYou are a helpful, tool-enabled assistant.")
        .add_tool(weather_tool)
        .add_mcp_server(McpServerType::stdio("npx -y @modelcontextprotocol/server-everything"))
        .add_mcp_server(McpServerType::streamable_http("http://localhost:8000/mcp"))
        .build()
        .await?;

    // 3. Use it!
    let greeting = agent.invoke("Say hello").await?;
    println!("Greeting: {}", greeting.content.unwrap_or_default());

    let forecast = agent.invoke("What is the current weather in Berlin?").await?;
    println!("Forecast: {}", forecast.content.unwrap_or_default());

    Ok(())
}
```

---

## API Reference

### `AgentBuilder`

* `set_model(model: impl Into<String>) -> Self`
  Choose the Ollama model identifier.

* `set_ollama_endpoint(url: impl Into<String>) -> Self`
  Base URL (default: `http://localhost`).

* `set_ollama_port(port: u16) -> Self`
  Port (default: `11434`).

* `set_system_prompt(prompt: impl Into<String>) -> Self`
  Initial system message to steer behaviour.

* `set_response_format(schema: impl Into<String>) -> Self`
  JSON Schema string for structured replies.

* `add_tool(tool: Tool) -> Self`
  Attach a prebuilt `Tool`.

* `add_mcp_server(server: McpServerType) -> Self`
  Discover tools via MCP.

* `build() -> Result<Agent, AgentBuildError>`
  Finalize and return an `Agent`.

### `Agent`

* `invoke(prompt: impl Into<String>) -> Result<Message, AgentError>`
  Send a user prompt, handle tool calls, and return the final message.

* **History** stored internally: sequence of system, user, model, and tool messages.

### `ToolBuilder`

* `new() -> Self`
  Start a fresh builder.

* `function_name(name: impl Into<String>) -> Self`
  Required: name of the tool in LLM schema.

* `function_description(desc: impl Into<String>) -> Self`
  Required: description for the tool.

* `add_property(key, type, desc) -> Self`
  Define an argument property.

* `add_required_property(key) -> Self`
  Mark a property as required.

* `executor(fn: AsyncToolFn) -> Self`
  Required: async executor closure.

* `build() -> Result<Tool, ToolBuilderError>`
  Validate and return a `Tool`.

---

## Contributing

We welcome your contributions to **Ragent**! Whether it’s bug fixes, new features, or improved docs, please:

1. Fork this repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m "Add feature"\`)
4. Push to your fork (`git push origin feature/my-feature`)
5. Open a Pull Request against `main`

Please ensure all tests pass (`cargo test`) and adhere to Rustfmt/style guidelines.

---

## License

Project is released under the MIT License. See [LICENSE](LICENSE) for details.
