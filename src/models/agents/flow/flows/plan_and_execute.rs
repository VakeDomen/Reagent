use reqwest::Response;
use serde_json;
use tracing::instrument;

use crate::{
    models::{
        flow::{
            invocation_flows::FlowFuture,
            util::{invocations::{call_model, invoke, invoke_with_tool_calls, invoke_without_tools}, request_generation::generate_llm_request_without_tools},
        },
        AgentError,
    },
    services::ollama::models::base::Message,
    Agent,
};


/// Runs a temporary, one-shot agent with a specific system prompt and returns the response content.
/// This is useful for sub-tasks within a larger flow, like planning or critiquing.
pub async fn run_sub_agent_once(
    main_agent: &Agent,
    system_prompt: String,
    user_prompt: String,
    json_format: bool,
) -> Result<String, AgentError> {
    let mut sub_agent = main_agent.clone();
    sub_agent.system_prompt = system_prompt;
    sub_agent.history = vec![
        Message::system(sub_agent.system_prompt.clone()),
        Message::user(user_prompt),
    ];

    if json_format {
        sub_agent.response_format = Some(serde_json::json!({
            "type": "array",
            "items": {
                "type": "string"
            }
        }));
    } else {
        // Inherit the main agent's format if not forcing JSON
        sub_agent.response_format = main_agent.response_format.clone();
    }

    // Sub-agents for planning/critique typically don't need tools.
    sub_agent.tools = None;

    let request = generate_llm_request_without_tools(&mut sub_agent).await?;
    let response = call_model(&sub_agent, request).await?;

    println!("RESP: {:#?}", response);

    Ok(response.message.content.unwrap_or_default())
}

const PLANNER_PROMPT: &str = "Your task is to create a detailed, step-by-step plan to solve the user's objective.

in the future theese will be the tools avalible to you: 

{tools}

First, carefully understand the objective.
Then, devise a plan by breaking down the entire task into smaller, executable sub-tasks.
Each step in the plan should be a single, clear action.
The final step of the plan must be to provide the final answer to the user's objective.
Do not add any superfluous or unnecessary steps.
Respond with a JSON-formatted list of strings, where each string is a single step in the plan.";

const REPLANNER_PROMPT: &str = "You are a planning agent responsible for adapting a plan based on the results of executed steps.
Your original objective was: {input}
Your original plan was: {plan}
You have already completed the following steps and observed their results:
{past_steps}

Based on the results, critically evaluate the plan. Update it by providing only the necessary remaining steps.
If the objective has been fully achieved, respond with an empty JSON list.
Do not repeat any steps that are already listed in the 'past_steps'.
Respond with a JSON-formatted list of strings representing the new plan.";

#[instrument(level = "debug", skip(agent, prompt))]
pub fn plan_and_execute_invoke<'a>(agent: &'a mut Agent, prompt: String) -> FlowFuture<'a> {
    Box::pin(async move {
        let max_turns = 5;
        let mut past_steps: Vec<(String, String)> = Vec::new();

        // 1. Initial Planning using a sub-agent
        agent.history.push(Message::user(prompt.clone()));

        let system_prompt = PLANNER_PROMPT
            .replace("{tools}", &format!("{:#?}", agent.get_compiled_tools().await?));
        let plan_content = run_sub_agent_once(
            agent, 
            system_prompt, 
            prompt.clone(), 
            true
        ).await?;
        
        let mut plan: Vec<String> = serde_json::from_str(&plan_content).map_err(|e| {
            AgentError::RuntimeError(format!("Planner failed to return valid JSON: {}", e))
        })?;
        let original_plan_str = format!("{:?}", plan);

        println!("PLAN:  {original_plan_str}");

        for _ in 0..max_turns {
            // 2. Check for termination
            if plan.is_empty() {
                break;
            }

            println!("PL {:#?}", plan);

            // 3. Execute the next step using the main agent
            let current_step = plan.remove(0);
            agent.history.push(Message::user(current_step.clone()));

            // The executor is a standard invocation that can use tools
            let _ = invoke_with_tool_calls(agent).await?;
            let response = invoke_without_tools(agent).await?;
            let observation = response.message.content.clone().unwrap_or_default();

            past_steps.push((current_step, observation));

            // 4. Re-plan using a sub-agent
            let past_steps_str = past_steps
               .iter()
               .map(|(step, result)| format!("Step: {}\nResult: {}", step, result))
               .collect::<Vec<_>>()
               .join("\n\n");

            let replanner_system_prompt = REPLANNER_PROMPT
               .replace("{input}", &prompt)
               .replace("{plan}", &original_plan_str)
               .replace("{past_steps}", &past_steps_str);

            let new_plan_content =
                run_sub_agent_once(agent, replanner_system_prompt, "".to_string(), true).await?;

            plan = serde_json::from_str(&new_plan_content).unwrap_or_default();
        }

        if let Some((_, final_result)) = past_steps.last() {
            Ok(Message::assistant(final_result.clone()))
        } else {
            Err(AgentError::RuntimeError(
                "Plan-and-Execute failed to produce a result.".into(),
            ))
        }
    })
}