# Reagent 🦀

Reagent is a Rust library for building, composing, and running LLM agents via Ollama. It provides a flexible and extensible framework for creating agents with custom tools, structured outputs, and complex invocation flows, designed with simplicity and composability in mind.

Currently only works with [Ollama](https://ollama.com/).

### Disclaimer

⚠️ **This library is experimental and created for educational purposes.** The API is under active development and is subject to breaking changes. It is not yet recommended for production use. Documentation is currently limited to the examples provided here and in the repository, as the library is not yet published to `crates.io`.

---

## Features

* **Tool Use**: Equip agents with local asynchronous Rust functions or connect to external tool providers.
* **Structured Output**: Force model responses into a specific JSON schema and deserialize them directly into your Rust structs.
* **Customizable Flows**: Replace the default agent invocation logic with your own custom functions or closures to implement patterns like ReAct, Plan-and-Execute, etc.
* **Event Notifications**: Subscribe to an agent's internal events (e.g., prompt generation, tool calls, final response) for logging, debugging, or streaming.
* **Prompt Templating**: Use simple or dynamic templates to construct prompts from variables and on-the-fly data sources.
* **MCP Integration**: Connect to external tool servers compliant with the Multi-Component Protocol (MCP) via SSE, stdio, or HTTP.

---

## Installation

Currently, `Reagent` is only available on GitHub. You can add it to your project by including the following in your `Cargo.toml`:

```toml
[dependencies]
reagent = { git = "https://github.com/VakeDomen/reagent.git" }
```

---

## Quick Start: A Simple Agent

Creating and interacting with an agent is straightforward. Use the `AgentBuilder` to configure your agent and `invoke_flow` to run it.

```rust
#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // 1. Create an agent using the builder pattern
    let mut agent = AgentBuilder::default()
        .set_model("qwen3:0.6b") // The only required setting
        .set_system_prompt("You are a helpful assistant.")
        .set_temperature(0.6)
        .set_num_ctx(2048) // lol
        .set_ollama_endpoint("http://localhost:2222")
        .build()
        .await?;

    // 2. Invoke the agent with a prompt
    let resp = agent.invoke_flow("How do I increase the context size in Ollama?").await?;
    println!("-> Agent: {}", resp.content.unwrap_or_default());

    // The agent maintains conversation history automatically
    let resp = agent.invoke_flow("What did you just say?").await?;
    println!("-> Agent: {}", resp.content.unwrap_or_default());

    // You can clear the history at any time (system prompt remains)
    agent.clear_history();

    Ok(())
}
```

---

## Core Concepts & Examples

For more examples, see [examples](https://github.com/VakeDomen/Reagent/tree/main/examples) folder.

### 1. Structured JSON Output

Ensure the model's output conforms to a specific JSON schema and deserialize it directly into your custom `struct`. This is ideal for predictable, machine-readable responses.

```rust
#[derive(Debug, Deserialize)]
struct MyWeatherOuput {
  windy: bool,
  temperature: i32,
  description: String
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    init_default_tracing();

    let mut agent = AgentBuilder::default()
        .set_model("qwen3:0.6b")
        .set_system_prompt("You make up weather info in JSON.")
        // Set the desired JSON schema for the response
        .set_response_format(r#"
            {
              "type":"object",
              "properties":{
                "windy":{"type":"boolean"},
                "temperature":{"type":"integer"},
                "description":{"type":"string"}
              },
              "required":["windy","temperature","description"]
            }
        "#)
        .build()
        .await?;
    
    // `invoke_flow_structured_output` automatically deserializes the JSON
    let resp: MyWeatherOuput = agent.invoke_flow_structured_output("What is the weather in Koper?").await?;
    println!("\n-> Agent: {:#?}", resp);

    Ok(())
}
```

### 2. Using Local Tools

You can give an agent access to local asynchronous Rust functions. The agent will intelligently decide when to call them based on the user's prompt.

```rust
#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    init_default_tracing();

    // 1. Define the tool's executor logic as an async closure
    let weather_exec: AsyncToolFn = {
        Arc::new(move |_model_args_json: Value| {
            Box::pin(async move {
                // Your logic goes here. For this example, we return a fixed JSON string.
                Ok(r#"{"temperature": 18, "description": "Partly cloudy"}"#.into())
            })
        })
    };

    // 2. Build the tool with a name, description, and argument schema
    let weather_tool = ToolBuilder::new()
        .function_name("get_current_weather")
        .function_description("Returns a weather forecast for a given location")
        .add_property("location", "string", "City name")
        .add_required_property("location")
        .executor(weather_exec)
        .build()?;

    // 3. Add the tool to the agent
    let mut agent = AgentBuilder::default()
        .set_model("qwen3:0.6b")
        .set_system_prompt("You are a helpful assistant.")
        .add_tool(weather_tool)
        .build()
        .await?;

    // The agent will use the tool when appropriate
    let resp = agent.invoke_flow("What is the current weather in Koper?").await?;
    println!("\n-> Agent: {}", resp.content.unwrap_or_default());

    Ok(())
}
```


### 3. Notifications & Streaming

You can receive real-time notifications about an agent's internal operations by creating a notification channel. This is useful for streaming tokens to the user, logging tool calls, and observing the agent's state.

```rust
#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Use `build_with_notification` to get a receiver channel
    let (mut agent, mut notification_reciever) = AgentBuilder::default()
        .set_model("qwen3:0.6b")
        .set_stream(true) // Optionally enable token streaming notifications
        .build_with_notification()
        .await?;

    // Spawn a task to listen for notifications
    let handle = tokio::spawn(async move {
        while let Some(msg) = notification_reciever.recv().await {
            match msg.content {
                NotificationContent::Token(token) => print!("{}", token),
                NotificationContent::ToolCallRequest(req) => println!("\n[Tool Call: {}]", req.tool_name),
                NotificationContent::Done(_, _) => println!("\n[Done]"),
                _ => {} // Handle other notification types
            }
        }
    });

    let _ = agent.invoke_flow("Say 'Hello, World!' and then tell me the weather in Paris.").await?;

    Ok(())
}
```

### 4. Custom Invocation Flows

For advanced use cases like ReAct or Plan-and-Execute, you can override the agent's default invocation logic with your own asynchronous function or closure. This gives you complete control over the interaction loop between the user, the model, and the tools.

```rust
// you can create own functions as the flows for invoking an agent
// when invoke_flow or invoke_flow_with_template is called,
// this is the function that will override the default flow if the 
// agent
// Box::pin to make the agent clonable. You can return Ok(Message)
// or Err(AgentError) from the fn
fn custom_flow<'a>(agent: &'a mut Agent, prompt: String) -> FlowFuture<'a> {
    Box::pin(async move {
        agent.history.push(Message::user(prompt));
        let mut last_response = None;
        // do your thing
        for _ in 0..agent.max_iterations.unwrap_or(1) {
            let response = invoke_without_tools(agent).await?;
            agent.history.push(response.message.clone());
            last_response = Some(response.message);
        }
        
        Ok(last_response.unwrap())
    })    
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    init_default_tracing();
    
    let mut agent = AgentBuilder::default()
        .set_model("qwen3:0.6b")
        .set_system_prompt("You are a helpful assistant.")
        .set_flow(Flow::Custom(custom_flow)) // Set the custom flow
        .set_max_iterations(3)
        .build()
        .await?;

    let resp = agent.invoke_flow("What is the meaning of life?").await?;
    println!("{:#?}", resp);

    Ok(())
}
```

### 5. Prompt Templating

Define reusable prompt templates and populate them with data at invocation time. You can use a simple `HashMap` or implement a `TemplateDataSource` for dynamic values like the current date.

```rust
#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    init_default_tracing();
    
    // Define a template with placeholders
    let template = Template::simple(r#"
        Your name is {{name}}.
        Answer the following question: {{question}}
    "#);

    let mut agent = AgentBuilder::default()
        .set_model("qwen3:0.6b")
        .set_template(template)
        .build()
        .await?;

    // Create a map with values for the placeholders
    let prompt_data = HashMap::from([
        ("name", "Gregor"),
        ("question", "What is your name?")
    ]);

    // Use `invoke_flow_with_template` to run the agent with the data
    let resp = agent.invoke_flow_with_template(prompt_data).await?;
    println!("-> Agent: {}", resp.content.unwrap_or_default());

    Ok(())
}
```
