
judge_prompt = """In reality, there are many tasks, which can be composed of multiple subtasks. There are often some relationships between these subtasks, such as some subtasks must be completed after other subtasks, that is, these subtasks depend on other subtasks.
Now I give you the main task and its corresponding two subtasks. You need to determine whether "subtask1" must be completed before "subtask2". If so, return [YES], otherwise return [NO]. Among them, "main_task" represents the main task, "subtask1" represents the first subtask, and "subtask2" represents the second subtask.
Here are some examples:

"main_task": make dinner
"subtask1": confirm menu
"subtask2": prepare ingredients
Answer: [YES]. For "prepare ingredients", we need to know what dishes we need to make before we can prepare the ingredients based on them, so we must first complete "confirm menu".

"main_task": make dinner
"subtask1": prepare ingredients
"subtask2": confirm menu
Answer: [NO]. For "prepare ingredients", we need to know what dishes we need to make before we can prepare the ingredients based on them, so we must first complete "confirm menu".

"main_task": build a deck around your pool
"subtask1": select materials
"subtask2": purchase the required materials
Answer: [YES]. When purchasing materials, you need to know which materials to buy, so we must first complete "select materials" and then complete "purchase the required materials"

"main_task": build a deck around your pool
"subtask1": purchase the required materials
"subtask2": select materials
Answer: [NO]. When purchasing materials, you need to know which materials to buy, so we must first complete "select materials" and then complete "purchase the required materials"

Now I will give you the main task and the corresponding two subtasks, and give your results. You must answer [YES] or [NO], and then give your reasons.
"main_task": (main_task)
"subtask1": (subtask1)
"subtask2": (subtask2)
Answer: 
"""


