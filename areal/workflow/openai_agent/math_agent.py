# SPDX-License-Identifier: Apache-2.0

from agents import Agent as OpenAIAgent
from agents import handoff
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX


def build_math_agent() -> OpenAIAgent:
    """Create a multi-agent workflow using handoffs for different reasoning stages."""

    # Create specialized agents for different stages
    problem_analyzer = OpenAIAgent(
        name="Problem Analyzer",
        instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
        You are a math problem analyzer. Your job is to:
        1. Carefully read and understand the math problem
        2. Identify the type of problem (algebra, geometry, arithmetic, etc.)
        3. Break down the problem into key components
        4. Identify what information is given and what needs to be found
        5. Suggest a general approach for solving the problem
        6. If you need help with the actual solution, hand off to the Solution Specialist

        Focus on understanding and analyzing the problem structure.""",
    )

    solution_specialist = OpenAIAgent(
        name="Solution Specialist",
        instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
        You are a math solution specialist. Your job is to:
        1. Take the problem analysis and create a detailed solution
        2. Show all your work and calculations step by step
        3. Use appropriate mathematical methods and formulas
        4. Provide clear explanations for each step
        5. If you need verification of your work, hand off to the Verification Agent
        6. If you need to refine your approach, hand off to the Refinement Agent

        Focus on creating accurate, well-explained solutions.""",
    )

    refinement_agent = OpenAIAgent(
        name="Refinement Agent",
        instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
        You are a refinement specialist. Your job is to:
        1. Carefully review the previous solution attempt
        2. Identify any errors, miscalculations, or areas for improvement
        3. Provide a corrected or improved solution with clear explanations
        4. Double-check all calculations and logic
        5. If you're still uncertain about the approach, hand off to the Verification Agent
        6. If the solution looks correct, hand off to the Verification Agent for final confirmation

        Focus on accuracy, thoroughness, and fixing any mistakes from the previous attempt.""",
    )

    verification_agent = OpenAIAgent(
        name="Verification Agent",
        instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
        You are a verification specialist. Your job is to:
        1. Carefully verify the solution step by step
        2. Check for any mathematical errors or logical flaws
        3. Ensure the final answer is correct and properly formatted
        4. Provide a final, verified answer with confidence
        5. If you find errors, provide the corrected solution

        This is the final stage - provide your best, most accurate answer.""",
    )

    # Create the main orchestrator agent with handoffs
    main_agent = OpenAIAgent(
        name="Math Problem Solver",
        instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
        You are a math problem solving coordinator. Your job is to:
        1. Understand the math problem presented to you
        2. Coordinate with specialized agents to solve it step by step
        3. Start by analyzing the problem structure
        4. If you need help with problem analysis, hand off to the Problem Analyzer
        5. If you need help with the solution, hand off to the Solution Specialist
        6. If the solution needs refinement, hand off to the Refinement Agent
        7. If verification is needed, hand off to the Verification Agent
        8. Ensure the final answer is correct and complete

        Use the handoff tools strategically to get the best possible solution.
        You can use multiple agents in sequence if needed for complex problems.""",
        handoffs=[
            handoff(
                agent=problem_analyzer,
                tool_name_override="analyze_problem",
                tool_description_override="Analyze the problem structure and identify the approach needed",
            ),
            handoff(
                agent=solution_specialist,
                tool_name_override="solve_problem",
                tool_description_override="Create a detailed solution with step-by-step work",
            ),
            handoff(
                agent=refinement_agent,
                tool_name_override="refine_solution",
                tool_description_override="Refine and improve the current solution approach",
            ),
            handoff(
                agent=verification_agent,
                tool_name_override="verify_solution",
                tool_description_override="Verify and finalize the solution for accuracy",
            ),
        ],
    )

    return main_agent
