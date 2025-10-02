use reagent_rs::{
    prelude::*, InvocationBuilder, ModelConfig, NotificationHandler, PromptConfig,
    StatefullPrebuild, StatelessPrebuild,
};
use serde_json::Value;
use std::{collections::HashMap, error::Error};
use tokio::sync::mpsc::Receiver;

const PLAN_AND_EXECUTE_SYSTEM_PROMPT: &str = r#"You are a **Chief Analyst and Reporter Agent**. Your job is to turn an execution log into a clear, well‑structured report for the end user.

    ### What you will receive
    * A conversation history in which
    1. **`User` messages** describe tasks that were executed.
    2. **`Assistant` messages** contain the raw results, observations, and any source URLs.

    ### Your final task
    Write **one cohesive report** that directly answers the user’s original objective.
    The final `User` message in the log restates that objective and tells you to begin.

    ---

    ## Report structure

    1. **Direct summary**
    Open with a single concise paragraph (no heading) that answers the core question.

    2. **Markdown body**
    Use headings (`##`), sub‑headings (`###`), **bold** for emphasis, and bulleted or numbered lists to organise the rest of the content.

    3. **Narrative from data**
    Weave the key findings into a logical story. Do **not** simply list results.

    4. **Citations**
    * Extract source URLs from the execution log.
    * Attach an inline citation immediately after each sourced fact, using a numbered link: `[1](http://example.com)`.
    * End the report with a `## References` section listing the full URLs in numeric order.

    *Citation example*

    > The programme coordinator is Dr. Jane Doe [1](http://example.com/dr‑jane‑doe).
    > Admission requires a completed bachelor’s degree [2](http://example.com/admission‑requirements).
    >
    > ## References
    > [1] http://example.com/dr‑jane‑doe
    > [2] http://example.com/admission‑requirements

    5. **Next steps**
    After the references, add `### Next Steps` with one or two helpful follow‑up questions or actions.

    ---

    ## Critical constraints

    * **Never mention your internal process or the tools used**; focus solely on providing the user with the
    information that was uncovered and the user might want to know.
    * **Base every statement strictly on the log content**.
    * Deliver the entire report as a single, self‑contained message.

    "#;

const PLANNER_SYSTEM_PROMPT: &str = r#"You are a meticulous Tactical Planner Agent. You will be given a high-level **strategy** and the original user **objective**. Your **sole purpose** is to convert that strategy into a detailed, step-by-step plan in a strict JSON format.

    **Your Task:**
    Based on the provided strategy, create a JSON object with a single key, "steps", whose value is an array of strings representing the plan. Do not add any explanations or introductory text. Your entire response must be only the JSON object.

    **Core Principle: The Executor is Blind**
    The Executor agent who runs these steps has **no knowledge** of the strategy or objective. Therefore, each step you create must be **100% self-contained and specific**, derived from the strategy and objective.

    **Rules for Plan Creation:**
    1.  **Translate Strategy to Tactics:** Convert each phase of the high-level strategy into one or more concrete, executable sub-tasks.
    2.  **Create Self-Contained Steps:** For each sub-task, formulate a precise, imperative instruction for the Executor. Embed all relevant keywords and context from the user's objective directly into the step's instruction.
    3.  **Specify Expected Output:** For each step, explicitly state what piece of information the Executor agent must find and return.
    4.  **Final Answer:** The very last step in the plan must **always** be: "Synthesize all the gathered information and provide the final, comprehensive answer to the user's objective."

    **Crucial Constraint: No Generic Steps**
    A step like `"Use query_memory to find relevant information"` is useless.
    - **Bad:** `Use rag_lookup to find information.`
    - **Good:** `Use rag_lookup to find information about FAMNIT's student exchange programs, partnerships, or support for students interested in visiting the Netherlands.`

    ---

    **Few-Shot Example:**

    **User Objective:** "Who was the monarch of the UK when the first person landed on the moon, and what was their full name?"

    **High-Level Strategy:** "The strategy will be to first establish the precise date of the initial moon landing. With that date confirmed, the next phase is to query historical records to identify who was the reigning monarch of the United Kingdom at that specific time. Finally, once the monarch is identified, a follow-up search will be needed to find their full, formal name to ensure accuracy."

    **Correct JSON Plan Output:**
    {
    "steps": [
        "Use the search_tool to find the exact date of the first moon landing and return the full date.",
        "Using the date from the previous step, use the search_tool to find who was the monarch of the United Kingdom at that specific time and return their common name.",
        "Using the name of the monarch from the previous step, use the search_tool to find their full given name and return that name.",
        "Synthesize the gathered information and provide the final answer to the user's objective."
    ]
    }
    "#;

const BLUEPRINT_SYSTEM_PROMPT: &str = r#"You are a Chief Strategist AI. Your role is to analyze a user's objective and devise a high-level, abstract strategy to achieve it. You do not create step-by-step plans or write code. Your output is a concise, natural language paragraph describing the strategic approach.

    **Your Thought Process:**
    1.  **Understand the Core Goal:** What is the fundamental question the user wants answered?
    2.  **Identify Key Information Areas:** What are the major pieces of information needed to reach the goal? (e.g., a date, a name, a location, a technical specification).
    3.  **Outline Logical Phases:** Describe the logical flow of the investigation in broad strokes. What needs to be found first to enable the next phase?
    4.  **Suggest General Capabilities:** Mention the *types* of actions needed (e.g., "search for historical data," "analyze technical documents," "cross-reference information") without specifying exact tool calls.

    **Output Rules:**
    -   Your entire response MUST be a single, natural language paragraph.
    -   **DO NOT** use JSON.
    -   **DO NOT** create a list of numbered or bulleted steps.
    -   **DO NOT** mention specific tool names like `search_tool` or `rag_lookup`.
    -   **DO NOT** call any tools yourself.

    ---
    **Example 1**

    **User Objective:** "Who was the monarch of the UK when the first person landed on the moon, and what was their full name?"

    **Correct Strategy Output:**
    The strategy will be to first establish the precise date of the initial moon landing. With that date confirmed, the next phase is to query historical records to identify who was the reigning monarch of the United Kingdom at that specific time. Finally, once the monarch is identified, a follow-up search will be needed to find their full, formal name to ensure accuracy.

    ---
    **Example 2**

    **User Objective:** "I'm planning to visit Netherlands next year. Can FAMNIT help me with anything?"

    **Correct Strategy Output:**
    The core strategy is to conduct a multi-pronged investigation into FAMNIT's resources. First, we need to explore official information about international student programs, focusing on any partnerships or exchange agreements with institutions in the Netherlands. Concurrently, we should check if specific academic programs have direct ties or collaborations. Finally, we can review internal knowledge bases for any documented precedents or support services related to student travel to that country.
    "#;

const REPLANNER_SYSTEM_PROMPT: &str = r#"You are an expert Re-Planner Agent. Your task is to analyze the progress made on a plan and create a new, revised plan to achieve the original objective. You will be given the original objective, the original plan, and a history of the steps that have already been executed along with their results.

**Core Principle: The Executor is Still Blind**
The Executor agent who runs your new plan still has **no knowledge** of the original objective or past steps. Therefore, every new step you create must be **100% self-contained and specific**, embedding all necessary context.

**Your Thought Process:**
1.  **Re-evaluate the Objective:** First, carefully re-read the original user objective to ensure you are still on track.
2.  **Analyze the Results:** Scrutinize the results from the `past_steps`. Did they succeed? Did they fail? Did they return unexpected information? What new facts have been established?
3.  **Refine and Enrich the Future Plan:** This is your most critical task. Look at the remaining steps in the original plan. If a result from a `past_step` provides concrete data (like a date, a name, a number), you **must** rewrite the future steps to directly include this new data. Replace generic placeholders like "the date from the previous step" with the actual, known information.
4.  **Assess Viability:** Based on the results and the newly enriched plan, decide if the plan is still sound.
    * If a step failed or the results indicate a dead end, you **must** formulate a new, alternative step to overcome the obstacle. Pivot the plan.
    * If the results have fully satisfied the user's objective, your new plan should be empty.

**Rules for the New Plan:**
1.  **Create Self-Contained and Enriched Steps:** Every step in your new plan must be a precise, imperative instruction with all necessary context and newly acquired data embedded.
2.  **Do Not Repeat Completed Steps:** Your new plan must only contain steps that have **not** yet been executed.
3.  **Output Format:** Your response **must** be a JSON object with a single `steps` key. If the objective is complete, the value should be an empty array.

---

**Few-Shot Example 1: Pivoting on Failure**

**Objective:** "Find the email address for the head of the Computer Science department at FAMNIT."
**Original Plan:** { "steps": ["Use the ask_staff_expert tool to find the name...", "Using the name from the previous step, use the ask_staff_expert tool to find the email...", "Synthesize..."] }
**Past Steps & Results:**
[ { "step": "Use the ask_staff_expert tool to find the name...", "result": "Execution Error: The tool 'ask_staff_expert' does not have information categorized by department leadership." } ]

**Correct New JSON Plan Output:**
{
  "steps": [
    "Use the get_web_page_content tool to search the official FAMNIT website for the 'Computer Science Department' page to identify and return the full name of the department head.",
    "Using the name of the department head found in the previous step, use the ask_staff_expert tool to find the email address for that person and return the email.",
    "Synthesize the gathered information and provide the final answer to the user's objective."
  ]
}

---

**Few-Shot Example 2: Enriching with New Data**

**Objective:** "Who was the monarch of the UK when the first person landed on the moon, and what was their full name?"
**Original Plan:**
{
  "steps": [
    "Use the search_tool to find the exact date of the first moon landing and return the full date.",
    "Using the date from the previous step, use the search_tool to find who was the monarch of the United Kingdom at that specific time and return their common name.",
    "Using the name of the monarch from the previous step, use the search_tool to find their full given name and return that name.",
    "Synthesize the gathered information and provide the final answer to the user's objective."
  ]
}
**Past Steps & Results:**
[
  {
    "step": "Use the search_tool to find the exact date of the first moon landing and return the full date.",
    "result": "The first moon landing occurred on July 20, 1969."
  }
]

**Correct New JSON Plan Output:**
{
  "steps": [
    "Using the now known date of July 20, 1969, use the search_tool to find who was the monarch of the United Kingdom at that specific time and return their common name.",
    "Using the name of the monarch from the previous step, use the search_tool to find their full given name and return that name.",
    "Synthesize the gathered information and provide the final answer to the user's objective."
  ]
}

**Example of a Completed Objective:**

**Past Steps & Results:**
[
  {
    "step": "Use the rag_lookup tool to find the email for the international student office at FAMNIT.",
    "result": "The email for the international office is international.office@famnit.upr.si."
  }
]

**Correct New JSON Plan Output:**
{
  "steps": []
}
"#;

const EXECUTOR_SYSTEM_PROMPT: &str = r#"You are given a task and a set of tools. Complete the task.
    Your response must be exhaustive. Hoever respond only with verifiable information that you have recieved int the
    context. If possible cite sources of data and provide references. Answer in markdown in the folowind structure:

    # Answer

    <include your answer>

    # Additional information

    <any additional information that might be usefull but has to be sourced from the context>

    "#;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    init_default_tracing();

    // build your agent
    let (mut agent, mut notification_reciever) =
        plan_and_execute().build_with_notification().await?;

    // build thread that will recieve notifications via the channel
    tokio::spawn(async move {
        while let Some(msg) = notification_reciever.recv().await {
            if let NotificationContent::Token(t) = msg.content {
                print!("{}", t.value)
            }
        }
    });

    // use it
    let _resp = agent.invoke_flow("Say hello").await?;
    let _resp = agent
        .invoke_flow("What is the current weather in Koper?")
        .await?;
    let _resp = agent.invoke_flow("What do you remember?").await?;

    Ok(())
}

pub fn plan_and_execute() -> AgentBuilder {
    // this is the builder for the top-level agent
    StatefullPrebuild::reply_without_tools()
        .set_temperature(0.7)
        .set_min_p(0.0)
        .set_top_p(0.8)
        .set_top_k(20)
        .set_max_iterations(3)
        .set_system_prompt(PLAN_AND_EXECUTE_SYSTEM_PROMPT)
        .set_flow(flow!(plan_and_execute_flow))
        .set_name("Statefull_prebuild-plan_and_execute")
}

pub async fn plan_and_execute_flow(
    agent: &mut Agent,
    prompt: String,
) -> Result<Message, AgentError> {
    // ------ setup before agent flow loops ------

    // history of the steps that were executed by the agent and the step results
    // note this is not message history, but histrory of the plan that the agent
    // decides to execute (step, result)

    // However top-level agent will retain the history that will look like:
    // system prompt + (steps, results) + summary response to user
    let mut past_steps: Vec<(String, String)> = Vec::new();

    // creating subagents
    // subagents are created on invocation and are therefore "stateless" inside
    // the top-level agent.
    let (mut blueprint_agent, blueprint_notification_channel) =
        create_blueprint_agent(agent).await?;
    let (mut planner_agent, planner_notification_channel) = create_planner_agent(agent).await?;
    let (mut replanner_agent, replanner_notification_channel) =
        create_replanner_agent(agent).await?;
    let (mut executor_agent, executor_notification_channel) = create_executor_agent(agent).await?;

    // you can forward notifications that one agent procuces to the other agent,
    // such that multi-agent flows have the same output from the top-level agent
    agent.forward_notifications(blueprint_notification_channel);
    agent.forward_notifications(planner_notification_channel);
    agent.forward_notifications(replanner_notification_channel);
    agent.forward_notifications(executor_notification_channel);

    // ------ here actual agent flow starts ------

    // 1. blueprint
    //
    // fist we build the draft (blueprint) of how to tackle the user problem
    // we do this by invoking the blueprint sub-agent
    let blueprint = blueprint_agent
        .invoke_flow_with_template(HashMap::from([
            ("tools", format!("{:#?}", agent.tools)),
            ("prompt", prompt.clone()),
        ]))
        .await?;

    // if for some reason the blueprint was not created, throw a runtime error
    let Some(blueprint) = blueprint.content else {
        return Err(AgentError::Runtime("Blueprint was not created".into()));
    };

    // 2. plan
    //
    // from the blueprint we attempt to create the step-by-step plan of the
    // how to solve the user task
    let plan_content = planner_agent
        .invoke_flow_with_template(HashMap::from([
            ("tools", format!("{:#?}", agent.tools)),
            ("prompt", blueprint),
        ]))
        .await?;

    // from the response of the planner agent extract the plan
    // the agent should return a list of strings (steps)
    let mut plan = get_plan_from_response(&plan_content)?;

    // loop
    for iteration in 0.. {
        // if max-iteration is set at the top-level agent, check for stopping
        if let Some(max_iterations) = agent.max_iterations {
            if iteration > max_iterations {
                break;
            }
        }

        // also stop if plan is empty (we completed all steps)
        if plan.is_empty() {
            break;
        }

        // extract what step we need to do
        let current_step = plan.remove(0);

        // put the step instruction to the overarching agent history (so the top-level agent remembers the step)
        agent.history.push(Message::user(current_step.clone()));

        // execute the step
        // for this we use the executor sub-agent with clean history every iteration
        let response = executor_agent.invoke_flow(current_step.clone()).await?;

        // top-level agent remembers the response (result of step)
        agent.history.push(response.clone());

        // also save the (step, result) to the past_steps
        let observation = response.content.clone().unwrap_or_default();
        past_steps.push((current_step, observation));

        // parse past steps to a string to pass to the next agent
        let past_steps_str = past_steps
            .iter()
            .map(|(step, result)| format!("Step: {step}\nResult: {result}"))
            .collect::<Vec<_>>()
            .join("\n\n");

        // use replaner to adapt the plan to executed steps and their results
        // the replanner also resets history on each iteration, so we pass the
        // "past_steps" to show histroical progress
        let new_plan_content = replanner_agent
            .invoke_flow_with_template(HashMap::from([
                ("tools", format!("{:#?}", agent.tools)),
                ("prompt", prompt.clone()),
                ("plan", format!("{plan:#?}")),
                ("past_steps", past_steps_str),
            ]))
            .await?;

        // parse the plan from response again and replace the current plan
        // with the new one
        plan = get_plan_from_response(&new_plan_content)?;
    }

    if past_steps.last().is_some() {
        // summarize the history and provide the user a nice answer to
        // the prompt
        // this is the only real invocation of the top-level agent
        // everything else is sub-agents
        agent.history.push(Message::user(prompt.to_string()));
        // let response = invoke_without_tools(agent).await?;
        let response = InvocationBuilder::default()
            .use_tools(false)
            .invoke_with(agent)
            .await?;

        agent
            .notify_done(true, response.message.content.clone())
            .await;
        Ok(response.message)
    } else {
        agent
            .notify_done(
                true,
                Some("Plan-and-Execute failed to produce a result.".into()),
            )
            .await;
        Err(AgentError::Runtime(
            "Plan-and-Execute failed to produce a result.".into(),
        ))
    }
}

fn get_plan_from_response(plan_response: &Message) -> Result<Vec<String>, AgentError> {
    // parse the Vec<String> (plan steps) from the agent response

    let original_plan_string = plan_response.content.clone().unwrap_or_default();

    let plan: Value = serde_json::from_str(&original_plan_string)
        .map_err(|e| AgentError::Runtime(format!("Planner failed to return valid JSON: {e}")))?;

    let plan = plan.get("steps").ok_or_else(|| {
        AgentError::Runtime("JSON object is missing the required 'steps' key.".to_string())
    })?;

    let plan: Vec<String> = serde_json::from_value(plan.clone()).map_err(|e| {
        AgentError::Runtime(format!(
            "The 'steps' key is not a valid array of strings: {e}"
        ))
    })?;

    Ok(plan)
}

pub async fn create_planner_agent(
    ref_agent: &Agent,
) -> Result<(Agent, Receiver<Notification>), AgentBuildError> {
    // extract configurations of the top-level agent
    let (client_config, model_config, prompt_config) = extract_configurations(ref_agent).await;

    // define planner sub-agent's template
    let template = Template::simple(
        r#"
    # These tools will be avalible to the executor agent:

    {{tools}}

    Users task to create a JSON plan for:

    {{prompt}}
    "#,
    );

    // planner will not use tools, so we use a prebuild that's
    // not allowed to use tools for the base
    StatelessPrebuild::reply_without_tools()
        // we transfer the settings set to the top-level agent
        .import_client_config(client_config)
        .import_model_config(model_config)
        .import_prompt_config(prompt_config)
        // set custom name of the sub-agent for
        // logging and notifications
        .set_name("Statefull_prebuild-plan_and_execute-planner")
        // set response format to be array of strings
        // [plan_steps]
        .set_response_format_str(
            r#"
        {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    }
                }
            },
            "required": ["steps"]
        }
        "#,
        )
        .set_system_prompt(PLANNER_SYSTEM_PROMPT)
        .set_template(template)
        // clear histroy every time we invoke the agent
        .set_clear_history_on_invocation(true)
        .build_with_notification()
        .await
}

pub async fn create_blueprint_agent(
    ref_agent: &Agent,
) -> Result<(Agent, Receiver<Notification>), AgentBuildError> {
    // extract configurations of the top-level agent
    let (client_config, model_config, prompt_config) = extract_configurations(ref_agent).await;

    // define blueprint sub-agent's template
    let template = Template::simple(
        r#"
    # These tools will later be avalible to the executor agent:

    {{tools}}

    Users task to create a JSON plan for:

    {{prompt}}
    "#,
    );

    // blueprint will not use tools, so we use a prebuild that's
    // not allowed to use tools for the base
    StatelessPrebuild::reply_without_tools()
        // we transfer the settings set to the top-level agent
        .import_client_config(client_config)
        .import_model_config(model_config)
        .import_prompt_config(prompt_config)
        // set custom name of the sub-agent for
        // logging and notifications
        .set_name("Statefull_prebuild-plan_and_execute-blueprint")
        .set_system_prompt(BLUEPRINT_SYSTEM_PROMPT)
        .set_template(template)
        // clear histroy every time we invoke the agent
        .set_clear_history_on_invocation(true)
        .build_with_notification()
        .await
}

pub async fn create_replanner_agent(
    ref_agent: &Agent,
) -> Result<(Agent, Receiver<Notification>), AgentBuildError> {
    // extract configurations of the top-level agent
    let (client_config, model_config, prompt_config) = extract_configurations(ref_agent).await;

    // define replanner sub-agent's template
    let template = Template::simple(
        r#"
    in the future theese will be the tools avalible to you:

    {{tools}}

    # Your original objective(user's task to complete) was:

    {{prompt}}

    # Your original plan was:

    {{plan}}

    # You have already completed the following steps and observed their results:

    {{past_steps}}

    "#,
    );

    // re-planner will not use tools, so we use a prebuild that's
    // not allowed to use tools for the base
    StatelessPrebuild::reply_without_tools()
        // we transfer the settings set to the top-level agent
        .import_client_config(client_config)
        .import_model_config(model_config)
        .import_prompt_config(prompt_config)
        // set custom name of the sub-agent for
        // logging and notifications
        .set_name("Statefull_prebuild-plan_and_execute-replanner")
        .set_system_prompt(REPLANNER_SYSTEM_PROMPT)
        .set_template(template)
        // set response format to be array of strings
        // [new_plan_steps]
        .set_response_format_str(
            r#"
        {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    }
                }
            },
            "required": ["steps"]
        }
        "#,
        )
        // clear histroy every time we invoke the agent
        .set_clear_history_on_invocation(true)
        .build_with_notification()
        .await
}

pub async fn create_executor_agent(
    ref_agent: &Agent,
) -> Result<(Agent, Receiver<Notification>), AgentBuildError> {
    let (client_config, model_config, prompt_config) = extract_configurations(ref_agent).await;

    AgentBuilder::default()
        // we transfer the settings set to the top-level agent
        .import_client_config(client_config)
        .import_model_config(model_config)
        .import_prompt_config(prompt_config)
        // set custom name of the sub-agent for
        // logging and notifications
        .set_name("Statefull_prebuild-plan_and_execute-task_executor")
        .set_system_prompt(EXECUTOR_SYSTEM_PROMPT)
        // clear histroy every time we invoke the agent
        .set_clear_history_on_invocation(true)
        .build_with_notification()
        .await
}

async fn extract_configurations(agent: &Agent) -> (ClientConfig, ModelConfig, PromptConfig) {
    let client_config = agent.export_client_config();
    let model_config = agent.export_model_config();
    let prompt_config = agent.export_prompt_config().await.unwrap_or_default();
    (client_config, model_config, prompt_config)
}
