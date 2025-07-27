use std::collections::HashMap;

use serde_json::Value;
use tokio::sync::mpsc::Receiver;
use tracing::instrument;

use crate::{
    models::{agents::flow::invocation_flows::{Flow, FlowFuture}, AgentBuildError, AgentError}, 
    prebuilds::stateless::StatelessPrebuild, 
    util::{invocations::invoke, templating::Template}, 
    Agent, AgentBuilder, Message, Notification
};



#[instrument(level = "debug", skip(agent, prompt))]
pub(crate )fn plan_and_execute_flow<'a>(agent: &'a mut Agent, prompt: String) -> FlowFuture<'a> {
    Box::pin(async move {
        let mut past_steps: Vec<(String, String)> = Vec::new();
        let max_turns = 5;
        
        let (mut blueprint_agent, blueprint_notification_channel) = create_blueprint_agent(&agent).await?;
        let (mut planner_agent, planner_notification_channel) = create_planner_agent(&agent).await?;
        let (mut replanner_agent, replanner_notification_channel) = create_replanner_agent(&agent).await?;
        let (mut executor_agent, executor_notification_channel) = create_single_task_agent(&agent).await?;

        agent.forward_notifications(blueprint_notification_channel);
        agent.forward_notifications(planner_notification_channel);
        agent.forward_notifications(replanner_notification_channel);
        agent.forward_notifications(executor_notification_channel);

        let blueprint = blueprint_agent.invoke_flow_with_template(HashMap::from([
            ("tools", format!("{:#?}", agent.tools)),
            ("prompt", prompt.clone())
        ])).await?;

        println!("Attempted to call tools: {:#?}", blueprint.tool_calls);

        let Some(blueprint) = blueprint.content else {
            return Err(AgentError::RuntimeError("Blueprint was not created".into()));
        };

        println!("BLUEPRINT: {}", blueprint);

        let plan_content = planner_agent.invoke_flow_with_template(HashMap::from([
            ("tools", format!("{:#?}", agent.tools)),
            ("prompt", blueprint)
        ])).await?;

        let mut plan = get_plan_from_response(&plan_content)?;
        
        println!("PLAN: {:#?}", plan);

        for _ in 0..max_turns {
            if plan.is_empty() {
                break;
            }


            let current_step = plan.remove(0);
            agent.history.push(Message::user(current_step.clone()));

            let response = executor_agent.invoke_flow(current_step.clone()).await?;
            agent.history.push(response.clone());
            let observation = response.content.clone().unwrap_or_default();
            past_steps.push((current_step, observation));

            
            let past_steps_str = past_steps
               .iter()
               .map(|(step, result)| format!("Step: {}\nResult: {}", step, result))
               .collect::<Vec<_>>()
               .join("\n\n");


            replanner_agent.clear_history();
            let new_plan_content = replanner_agent.invoke_flow_with_template(HashMap::from([
                ("tools", format!("{:#?}", agent.tools)),
                ("prompt", prompt.clone()),
                ("plan", format!("{:#?}", plan)),
                ("past_steps", past_steps_str),
            ])).await?;
            plan = get_plan_from_response(&new_plan_content)?;
        }


        if let Some(_) = past_steps.last() {

            agent.history.push(Message::user(format!("{}", prompt)));
            let response = invoke(agent).await?;

            agent.notify(crate::NotificationContent::Done(true)).await;
            Ok(response.message)
        } else {
            agent.notify(crate::NotificationContent::Done(false)).await;
            Err(AgentError::RuntimeError(
                "Plan-and-Execute failed to produce a result.".into(),
            ))
        }
    })    
}


impl AgentBuilder {
    pub fn plan_and_execute() -> AgentBuilder {

        let system_prompt = r#"You are a Chief Analyst and Reporter Agent. Your primary function is to transform a log of technical steps and results into a comprehensive, well-structured, and easy-to-read report for the end-user.

The conversation history you will receive is an execution log detailing:
1.  The user's original high-level objective.
2.  A sequence of specific tasks that were executed (`User` messages).
3.  The raw results and observations from each task (`Assistant` messages).

**Your Final Task: Create a Detailed Report**
Your final response must be a comprehensive report that directly answers the user's original objective, which is repeated as the final message in the history. This is your signal to begin.

**Report Structure and Formatting Rules:**
1.  **Begin with a Direct, Conversational Response:** Start your report by directly answering the user's core question in a natural, conversational tone. **Do not use a generic heading like "Direct Answer" for this opening.**
2.  **Use Extensive Markdown:** After the initial response, structure the rest of the report using markdown. Use headings (`##`), subheadings (`###`), bold text for emphasis on key terms, and bulleted or numbered lists to break down information.
3.  **Elaborate on the Findings:** Do not just state the final answer. Elaborate on the key information discovered during the execution process. Create separate sections for different aspects of the findings to build a comprehensive picture.
4.  **Create a Narrative:** Weave the key findings from the log into a logical narrative. You can explain *what* was found at major stages to build your answer (e.g., "Initial research into the moon landing date confirmed it was July 20, 1969. Subsequent searches for the UK monarch at that time revealed...").
5.  **Do Not Mention the Process Itself:** Crucially, **do not** talk about the "plan," "steps," or "tools" (e.g., do not say "The first step was to use the search tool"). Instead, focus on the *information* that was uncovered.

**CRITICAL CONSTRAINTS:**
-   Your entire response must be a single, cohesive report.
-   You **must not** attempt to call any tools or re-execute tasks.
-   Base your report **exclusively** on the facts present in the conversation log."#;

        AgentBuilder::default()
            .set_temperature(0.)
            .set_system_prompt(system_prompt)
            .set_flow(Flow::Custom(plan_and_execute_flow))
            .set_name("Statefull_prebuild-plan_and_execute")
    }
}

fn get_plan_from_response(plan_response: &Message) -> Result<Vec<String>, AgentError> {
    let original_plan_string = plan_response.content.clone().unwrap_or_default();
    let plan: Value = serde_json::from_str(&original_plan_string).map_err(|e| {
        AgentError::RuntimeError(format!("Planner failed to return valid JSON: {}", e))
    })?;

    let plan = plan.get("steps").ok_or_else(|| {
        AgentError::RuntimeError("JSON object is missing the required 'steps' key.".to_string())
    })?;

    let plan: Vec<String> = serde_json::from_value(plan.clone()).map_err(|e| {
        AgentError::RuntimeError(format!("The 'steps' key is not a valid array of strings: {}", e))
    })?;

    Ok(plan)
}

pub async fn create_planner_agent(ref_agent: &Agent) -> Result<(Agent, Receiver<Notification>), AgentBuildError> {
    let system_prompt = r#"You are a meticulous Tactical Planner Agent. You will be given a high-level **strategy** and the original user **objective**. Your **sole purpose** is to convert that strategy into a detailed, step-by-step plan in a strict JSON format.

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

    let template = Template::simple(r#"
    # These tools will be avalible to the executor agent: 

    {{tools}}

    Users task to create a JSON plan for: 

    {{prompt}}
    "#);

    let mut builder = StatelessPrebuild::reply_without_tools()
        .set_name("Statefull_prebuild-plan_and_execute-planner")
        .set_model(ref_agent.model.clone())
        .set_ollama_endpoint(ref_agent.ollama_client.base_url.clone());

    if let Some(v) = ref_agent.temperature { builder = builder.set_temperature(v); }
    if let Some(v) = ref_agent.top_p { builder = builder.set_top_p(v); }
    if let Some(v) = ref_agent.presence_penalty { builder = builder.set_presence_penalty(v); }
    if let Some(v) = ref_agent.frequency_penalty { builder = builder.set_frequency_penalty(v); }
    if let Some(v) = ref_agent.num_ctx { builder = builder.set_num_ctx(v); }
    if let Some(v) = ref_agent.repeat_last_n { builder = builder.set_repeat_last_n(v); }
    if let Some(v) = ref_agent.repeat_penalty { builder = builder.set_repeat_penalty(v); }
    if let Some(v) = ref_agent.seed { builder = builder.set_seed(v); }
    if let Some(v) = &ref_agent.stop { builder = builder.set_stop(v.clone()); }
    if let Some(v) = ref_agent.num_predict { builder = builder.set_num_predict(v); }
    if let Some(v) = ref_agent.top_k { builder = builder.set_top_k(v); }
    if let Some(v) = ref_agent.min_p { builder = builder.set_min_p(v); }
    if let Some(v) = &ref_agent.local_tools { for t in v {builder = builder.add_tool(t.clone());}}
    if let Some(v) = &ref_agent.mcp_servers { for t in v {builder = builder.add_mcp_server(t.clone());}}
    if !ref_agent.name.is_empty() { builder = builder.set_name(format!("{}-planner", ref_agent.name)) }
    builder = builder.strip_thinking(ref_agent.strip_thinking);




    builder
        .set_response_format(r#"
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
        "#)
        .set_system_prompt(system_prompt)
        .set_template(template)
        .set_clear_history_on_invocation(true)
        .set_model(ref_agent.model.clone())
        .build_with_notification()
        .await
}



pub async fn create_blueprint_agent(ref_agent: &Agent) -> Result<(Agent, Receiver<Notification>), AgentBuildError> {
    let system_prompt = r#"You are a Chief Strategist AI. Your role is to analyze a user's objective and devise a high-level, abstract strategy to achieve it. You do not create step-by-step plans or write code. Your output is a concise, natural language paragraph describing the strategic approach.

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

    let template = Template::simple(r#"
    # These tools will later be avalible to the executor agent: 

    {{tools}}

    Users task to create a JSON plan for: 

    {{prompt}}
    "#);

    let mut builder = StatelessPrebuild::reply_without_tools()
        .set_name("Statefull_prebuild-plan_and_execute-blueprint")
        .set_model(ref_agent.model.clone())
        .set_ollama_endpoint(ref_agent.ollama_client.base_url.clone());

    if let Some(v) = ref_agent.temperature { builder = builder.set_temperature(v); }
    if let Some(v) = ref_agent.top_p { builder = builder.set_top_p(v); }
    if let Some(v) = ref_agent.presence_penalty { builder = builder.set_presence_penalty(v); }
    if let Some(v) = ref_agent.frequency_penalty { builder = builder.set_frequency_penalty(v); }
    if let Some(v) = ref_agent.num_ctx { builder = builder.set_num_ctx(v); }
    if let Some(v) = ref_agent.repeat_last_n { builder = builder.set_repeat_last_n(v); }
    if let Some(v) = ref_agent.repeat_penalty { builder = builder.set_repeat_penalty(v); }
    if let Some(v) = ref_agent.seed { builder = builder.set_seed(v); }
    if let Some(v) = &ref_agent.stop { builder = builder.set_stop(v.clone()); }
    if let Some(v) = ref_agent.num_predict { builder = builder.set_num_predict(v); }
    if let Some(v) = ref_agent.top_k { builder = builder.set_top_k(v); }
    if let Some(v) = ref_agent.min_p { builder = builder.set_min_p(v); }
    if let Some(v) = &ref_agent.local_tools { for t in v {builder = builder.add_tool(t.clone());}}
    if let Some(v) = &ref_agent.mcp_servers { for t in v {builder = builder.add_mcp_server(t.clone());}}
    if !ref_agent.name.is_empty() { builder = builder.set_name(format!("{}-blueprint", ref_agent.name)) }
    builder = builder.strip_thinking(ref_agent.strip_thinking);




    builder
        .set_system_prompt(system_prompt)
        .set_template(template)
        .set_model(ref_agent.model.clone())
        .set_clear_history_on_invocation(true)
        .build_with_notification()
        .await
}



pub async fn create_replanner_agent(ref_agent: &Agent) -> Result<(Agent, Receiver<Notification>), AgentBuildError> {
    let system_prompt = r#"You are an expert Re-Planner Agent. Your task is to analyze the progress made on a plan and create a new, revised plan to achieve the original objective. You will be given the original objective, the original plan, and a history of the steps that have already been executed along with their results.

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

    let template = Template::simple(r#"
    in the future theese will be the tools avalible to you: 

    {{tools}}

    # Your original objective(user's task to complete) was: 

    {{prompt}}
    
    # Your original plan was: 
    
    {{plan}}
    
    # You have already completed the following steps and observed their results:
    
    {{past_steps}}

    "#);

    let mut builder = StatelessPrebuild::reply_without_tools()
        .set_name("Statefull_prebuild-plan_and_execute-replanner")
        .set_model(ref_agent.model.clone())
        .set_ollama_endpoint(ref_agent.ollama_client.base_url.clone());

    if let Some(v) = ref_agent.temperature { builder = builder.set_temperature(v); }
    if let Some(v) = ref_agent.top_p { builder = builder.set_top_p(v); }
    if let Some(v) = ref_agent.presence_penalty { builder = builder.set_presence_penalty(v); }
    if let Some(v) = ref_agent.frequency_penalty { builder = builder.set_frequency_penalty(v); }
    if let Some(v) = ref_agent.num_ctx { builder = builder.set_num_ctx(v); }
    if let Some(v) = ref_agent.repeat_last_n { builder = builder.set_repeat_last_n(v); }
    if let Some(v) = ref_agent.repeat_penalty { builder = builder.set_repeat_penalty(v); }
    if let Some(v) = ref_agent.seed { builder = builder.set_seed(v); }
    if let Some(v) = &ref_agent.stop { builder = builder.set_stop(v.clone()); }
    if let Some(v) = ref_agent.num_predict { builder = builder.set_num_predict(v); }
    if let Some(v) = ref_agent.top_k { builder = builder.set_top_k(v); }
    if let Some(v) = ref_agent.min_p { builder = builder.set_min_p(v); }
    if let Some(v) = &ref_agent.local_tools { for t in v {builder = builder.add_tool(t.clone());}}
    if let Some(v) = &ref_agent.mcp_servers { for t in v {builder = builder.add_mcp_server(t.clone());}}
    if !ref_agent.name.is_empty() { builder = builder.set_name(format!("{}-replanner", ref_agent.name)) }
    builder = builder.strip_thinking(ref_agent.strip_thinking);


    builder
        .set_system_prompt(system_prompt)
        .set_template(template)
        .set_model(ref_agent.model.clone())
        .set_response_format(r#"
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
        "#)
        .set_clear_history_on_invocation(true)
        .build_with_notification()
        .await
}



pub async fn create_single_task_agent(ref_agent: &Agent) -> Result<(Agent, Receiver<Notification>), AgentBuildError> {
    let system_prompt = r#"You are given a task and a set of tools. Complete the task.
    You may use the tools if they are heplful. Once you have a response for the task ready, wrap the
    final response to the user in the <final>response</final>"#;

    let mut builder = AgentBuilder::default()
        .set_name("Statefull_prebuild-plan_and_execute-task_executor")
        .set_model(ref_agent.model.clone())
        .set_ollama_endpoint(ref_agent.ollama_client.base_url.clone());

    if let Some(v) = ref_agent.temperature { builder = builder.set_temperature(v); }
    if let Some(v) = ref_agent.top_p { builder = builder.set_top_p(v); }
    if let Some(v) = ref_agent.presence_penalty { builder = builder.set_presence_penalty(v); }
    if let Some(v) = ref_agent.frequency_penalty { builder = builder.set_frequency_penalty(v); }
    if let Some(v) = ref_agent.num_ctx { builder = builder.set_num_ctx(v); }
    if let Some(v) = ref_agent.repeat_last_n { builder = builder.set_repeat_last_n(v); }
    if let Some(v) = ref_agent.repeat_penalty { builder = builder.set_repeat_penalty(v); }
    if let Some(v) = ref_agent.seed { builder = builder.set_seed(v); }
    if let Some(v) = &ref_agent.stop { builder = builder.set_stop(v.clone()); }
    if let Some(v) = ref_agent.num_predict { builder = builder.set_num_predict(v); }
    if let Some(v) = ref_agent.top_k { builder = builder.set_top_k(v); }
    if let Some(v) = ref_agent.min_p { builder = builder.set_min_p(v); }
    if let Some(v) = &ref_agent.local_tools { for t in v {builder = builder.add_tool(t.clone());}}
    if let Some(v) = &ref_agent.mcp_servers { for t in v {builder = builder.add_mcp_server(t.clone());}}
    if !ref_agent.name.is_empty() { builder = builder.set_name(format!("{}-executor", ref_agent.name)) }
    builder = builder.strip_thinking(true);


    builder
        .set_system_prompt(system_prompt)
        .set_model(ref_agent.model.clone())
        .set_stopword("</final>")
        .set_max_iterations(10)
        .set_clear_history_on_invocation(true)
        .build_with_notification()
        .await
}


