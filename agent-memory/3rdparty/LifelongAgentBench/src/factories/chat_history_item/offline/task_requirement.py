from typing import Mapping

from src.typings import TaskName


TASK_REQUIREMENT_DICT: Mapping[TaskName, str] = {
    TaskName.DB_BENCH: """I will ask you a question, then you should help me operate a MySQL database with SQL to answer the question.
You have to explain the problem and your solution to me and write down your thoughts.
After thinking and explaining thoroughly, every round you can choose one of the two actions: Operation or Answer.

To do operation, the format should be like this:
Action: Operation
```sql
SELECT * FROM table WHERE condition;
```
You MUST put SQL in markdown format without any other comments. Your SQL should be in one line.
I will use "Action: Operation" literally to match your SQL.
Every time you can only execute one SQL statement. I will only execute the statement in the first SQL code block. Every time you write a SQL, I will execute it for you and give you the output.
If the SQL is not executed successfully, the response will be the error message.
Otherwise, the response will be the raw MySQL response.
For SELECT queries, the response will be the result of the query, such as [(1, 'John Doe', 'HR'), (2, 'Jane Smith', 'IT'), ...], where each tuple represents a row and the elements are the values of the columns in the row.
For SQL such as INSERT, UPDATE, and DELETE, the response will be an empty list [] indicating that the SQL was executed successfully.

If you have obtain the answer by interacting with the database, and you MUST commit your final answer using the format like this:
Action: Answer
Final Answer: [(1, 'John Doe', 'HR'), (2, 'Jane Smith', 'IT'), ...]
DO NOT write this pattern unless you are sure about your answer. I expect an accurate and correct answer.
Your answer should be accurate. Your answer must be exactly the same as the correct answer.
If the question is about modifying the database, then after done operation, your answer field can be anything.
If the question is about querying the database, then after done operation, your answer field should be the result of the query.
We note that the column names will not be displayed in the result, and you need to ensure both the orders of the columns and rows are correct.
If your response cannot match any pattern I mentioned earlier, you will be judged as FAIL immediately.
Once you commit your answer or the number of rounds reaches 3, the task will be finished and the system will judge whether you pass the task or not.""",
    TaskName.OS_INTERACTION: """I will provide you with a task to perform on a Linux (Ubuntu) system. Your objective is to complete the task by executing the appropriate Bash commands.

### Interaction Rules:
1. **Thorough Analysis and Reasoning**:
    - Before performing any action, carefully analyze the task and explain your thought process.
    - Include a detailed explanation of the logic behind your choice of commands and approach.

2. **Action Choices**:
   - At the end of your reasoning, select **one and only one action** for each turn.
     - **"bash"**: When you need to execute a command or perform an operation, provide the corresponding Bash code. Structure your response as:
        Act: bash
        ```bash
        # Your Bash command(s) here
        ```
     - **"finish"**: When the task is complete and no further action is required, conclude with:
        Act: finish

3. **Other Guidelines**:
    - I will use "Act: bash" and "Act: finish" literally to determine whether your action is to execute commands or conclude the task.
    - Use the provided format accurately and consistently.
    - Ensure all Bash commands are compatible with Linux (Ubuntu) systems.
    - Avoid interactive operations (e.g., read, readline) in your Bash commands.

4. **Task Completion**:
    - The task will conclude either when you select the "finish" action or when the number of rounds reaches 5.
    - The system will evaluate your performance to determine if the task was successfully completed.""",
    TaskName.KNOWLEDGE_GRAPH: """You are an intelligent agent tasked with answering questions by querying a knowledge base. Your goal is to efficiently retrieve and process information using the tools provided.

### Interaction Rules:
    - Before performing any action, carefully analyze the task and explain your thought process.
    - Include a detailed explanation of the logic behind your choice of commands and approach.
    - Perform **only one action per turn** or provide the **Final Answer**.
    - You can only choose one action from the available set at each turn.
    - Adhere strictly to the specified formats for each action.

### Input Guidelines
- **Variable**: A result cached from previous actions (e.g., #0). It can either be either an entity or a set of entities.

### Available Actions
1. **get_relations(var: Variable | str) -> list of relations**
    - Retrieves all relations connected to the input str or Variable.
    - **Input**: str or Variable (e.g., Barack Obama or #0).
    - **Example**: Action: get_relations(Barack Obama) or Action: get_relations(#0).

2. **get_neighbors(var: Variable | str, relation: str) -> Variable**
    - Retrieves entities connected to the input Variable or str via the specified relation.
    - **Input**: Variable or str and a relation from get_relations().
    - **Prerequisite**: Use get_relations() to find the set of viable relations.
    - **Example**: Action: get_neighbors(#0, people.person.profession) or Action: get_neighbors(Barack Obama, people.person.profession).

3. **intersection(var1: Variable, var2: Variable) -> Variable**
    - Finds the intersection of two Variables containing entities of the same type.
    - **Input**: Two Variables (e.g., #0 and #1).
    - **Example**: Action: intersection(#0, #1).

4. **get_attributes(var: Variable) -> list of attributes**
    - Retrieves numerical attributes associated with the input Variable. 
    - **Input**: Variable (e.g., #0).
    - **Example**: Action: get_attributes(#0).

5. **argmax(var: Variable, attribute: str) -> Variable**
    - Returns the entity with the highest value for the specified attribute in the Variable.
    - **Prerequisite**: Use get_attributes() to identify valid attributes.
    - **Example**: Action: argmin(#1,time.event.end_date).

6. **argmin(var: Variable, attribute: str) -> Variable**
    - Returns the entity with the lowest value for the specified attribute in the Variable.
    - **Prerequisite**: Use get_attributes() to identify valid attributes.
    - **Example**: Action: argmin(#1,time.event.end_date).

7. **count(var: Variable) -> Variable**
    - Returns a Variable representing the number of entities in the input Variable.
    - **Example**: Action: count(#0).

### Final Answer Format
Submit your final answer in the following format:
    - Final Answer: #id (e.g., Final Answer: #3).

### Output Rules
1. The output of get_neighbors, intersection, argmax, and argmin is a **Variable**, represented as an ID (e.g., #0, #1).
    - Variables are sequentially numbered starting from #0.
    - Variables returned by **get_relations**, **get_neighbors**, **intersection**, **get_attributes** can be used as inputs for subsequent actions or as final answers.
    - Variables returned by **argmax**, **argmin**, and **count** can only be used as final answers.
2. The output of get_relations, get_attributes is direct and not stored as a Variable.
3. Only one variable can be submitted as a final answer.

### Task Completion
    - The task will conclude either when you select the "finish" action or when the number of rounds reaches 15.
    - The system will evaluate your performance to determine if the task was successfully completed.
""",
}
