# memp/agent/prompts.py

# This part is static during an episode.
SYSTEM_PROMPT = """Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish.
For each of your turn, you will be given the observation of the last turn. You should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:"Thought: your thoughts.\nAction: your next action".

The available actions are:
1. go to {recep}
2. take {obj} from {recep}
3. put {obj} in/on {recep}
4. open {recep}
5. close {recep}
6. use {obj}
7. clean {obj} with {recep}
8. heat {obj} with {recep}
9. cool {obj} with {recep}
where {obj} and {recep} correspond to objects and receptacles.
After your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output "Nothing happened", that means the previous action is invalid and you should try more options.

Your response should use the following format:
Thought: <your thoughts>
Action: <your next action>"""


# This template is for the user's message when memories are found.
WITH_MEMORY_PROMPT = """**Primary Goal:**
{task_description}

**Archived Memories (from similar past tasks):**
{retrieved_memories}

**Current Task Progress (recent steps):**
{history}
"""

# This template is for the user's message when no memories are found.
ZERO_SHOT_PROMPT = """**Primary Goal:**
{task_description}

**Archived Memories (from similar past tasks):**
No relevant memories were found. You must rely on your general knowledge.

**Current Task Progress (recent steps):**
{history}
"""


FEW_SHOT_PROMPT_SYSTEM = """
**Instructional Examples (from a manual):**
Here is an example of how to solve the task:
--- BEGIN EXAMPLES ---
{few_shot_examples}
--- END EXAMPLES ---

**Archived Memories (from your own past experiences):**
{retrieved_memories}
"""

FEW_SHOT_PROMPT_USER = """**Primary Goal:**
{task_description}
**Current Task Progress (recent steps):**
{history}
"""

# We can simplify the other prompts or keep them for ablation studies
WITH_MEMORY_PROMPT = """**Primary Goal:**
{task_description}

**Archived Memories (from your own past experiences):**
{retrieved_memories}

**Current Task Progress (recent steps):**
{history}
"""