
system_prompt_has_dependeny = """In reality, there are many tasks, which can be composed of multiple subtasks. Some of these subtasks have dependencies, that is, some tasks can only be completed after some other tasks are completed. But some tasks are not necessarily related, that is, these subtasks can be executed at the same time. Now I give you a main task, and you need to generate some subtasks with dependencies. "subtask1" means the task that needs to be completed first, and "subtask2" means the task that needs to be completed after "subtask1". Generate as many subtask pairs as possible. You need to make sure that the data you generate is completely correct. You need to return your answer in json format.
Here are some examples

"main_task": Create a business plan
[{"subtask1": "conduct market research", "subtask2": "identify target audience"}, {"subtask1": "define business goals", "subtask2": "outline operational plan"}, {"subtask1": "paint or finish the shelf", "subtask2": "organize items on the shelf"}]

"main_task": Build a simple wall shelf
[{"subtask1": "decide the location for the shelf", "subtask2": "mark the wall"}, {"subtask1": "measure and cut the shelf", "subtask2": "assemble shelf components"}]

Now give the task, which is "main_task".
"main_task": 
"""


system_prompt_no_relevant = """In reality, there are many tasks, which can be composed of multiple subtasks. Some of these subtasks have dependencies, that is, some tasks can only be completed after some other tasks are completed. But some tasks have no necessary connection, that is, these subtasks can be executed at the same time. Now I give you a main task, and you need to generate some subtasks without dependencies. That is to say, two subtasks can be executed at the same time without some dependencies. Generate as many subtask pairs as possible. You need to make sure that the data you generate is completely correct. You need to return your answer in json format.
Here are some examples

"main_task": Create a business plan
Answer: [{"subtask1": "conduct market research", "subtask2": "analyze competition"}, {"subtask1": "outline operational plan", "subtask2": "create organizational structure"}]

"main_task": Carving a jade jewelry piece
Answer: [{"subtask1": "create design concept", "subtask2": "obtain jade stone"}, {"subtask1": "create design concept", "subtask2": "prepare jade stone"}, {"subtask1": "assemble carving tools","subtask2": "prepare jade stone"}]

Now give the task, which is "main_task".
"main_task" :
"""
