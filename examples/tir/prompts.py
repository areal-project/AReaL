"""
Prompt templates for TIR (Tool-Integrated Reasoning) workflow.
"""

SYSTEM_PROMPT = """
You are a helpful assistant that can use tools to help the user.
You can use the following tools:
{tool_descriptions}
When you invoke a tool in your response, the tool's output will be immediately obtained and placed within the output``` ``` tags. Then, you continue answering based on the tool's output. Depending on the parameters you provide for the invocation, the tool's invocation may fail. You can invoke the tool multiple times in your response.
You should use the tools to help the user to solve the problem whenever possible.
Please reason step by step, and put your final answer within \\boxed{{}}.
"""

BASE_MODEL_PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant answers it. The Assistant analyzes the given question and information in the mind, retains important relevant information, calls multiple tools to find get necessary information, and provides the user with the answer.
The reasoning processes are enclosed within <think> </think>.
The available tools are:
{tool_descriptions}

Finally, the Assistant provides answer within \\boxed{{}}., i.e. \\boxed{{4}}.

User:
{question}

Assistant:
<think>"""

TORL_PROMPT = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it.\nUser:Please integrate natural language reasoning with programs to solve the problem blow, and put your final answer within \\boxed{{}}..\n{prompt}\nAssistant:"

ANSWER = r"\boxed{.*?}"
