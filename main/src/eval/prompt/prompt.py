import json

prompt_combine_small_and_LLM_1 = """In reality, there are many complex tasks, which can be broken down into many subtasks to solve. These subtasks often have dependencies (some subtasks must wait for other subtasks to complete before they can start). Now I give you the task and the corresponding subtasks. You need to construct them into a tree based on the dependencies between the subtasks so that some subtasks can run in parallel. You need to increase the parallelism of the subtasks as much as possible, that is, the phenomenon that more subtasks depend on the same subtask occurs more frequently. You need to return your answer in JSON format: "[{"subtask": ... "dependencies": [....]}, ......]", "subtask" represents the current subtask, and the subtasks appearing in "dependencies" need to be completed before the current subtask.
Now I give you a rough idea of the dependencies, most of which are correct, but some are wrong. You need to modify them based on your own knowledge and give the final answer.

Here are some examples:
Example 1:
task: make dinner
subtasks: ["confirm menu", "wash rice", "set up the table", "prepare ingredients", "cut vegetables", "clean and slicing meat", "make soup", "prepare dishes", "prepare dishes", "cook rice", "place the plate", "place bowl and chopsticks", "serve the food"]
Answer: [{"subtask": "confirm menu", "dependencies": ["make dinner"]}, {"subtask": "wash rice", "dependencies": ["make dinner"]}, {"subtask": "set up the table", "dependencies": ["make dinner"]}, {"subtask": "prepare ingredients", "dependencies": [""confirm menu"]}, {"subtask": "cut vegetables", "dependencies": ["prepare ingredients"]}, {"subtask": "clean and slicing meat": , "dependencies": ["prepare ingredients"]}, {"subtask": "make soup", "dependencies": ["prepare ingredients"]}, {"subtask": "prepare dishes" , "dependencies": ["cut vegetables", "clean and slicing meat"]}, {"subtask": "cook rice", "dependencies": ["wash rice"]}, {"subtask": "place the plate", "dependencies": ["set up the table"]}, {"subtask": "place bowl and chopsticks", "dependencies": ["set up the table"]}, {"subtask": "serve the food", "dependencies": ["prepare dishes", "make soup", "cook rice", "place the plate", "place bowl and chopsticks"]}]

Example 2:
task: build a deck around your pool
subtasks: ["obtain building permits", "determine the shape and size of your deck", "select materials", "prepare relevant tools", "purchase the required materials", "clean the site", "excavate the foundation pit", "build the deck frame", "install foundation supports(such as columns)", "install horizontal support beams on the frame", "lay deck planks", "install guardrail", "clean and decorate deck"]
Answer: [{"subtask": "obtain building permits", "dependencies": ["build a deck around your pool"]}, {"subtask": "determine the shape and size of your deck", "dependencies": ["build a deck around your pool"]}, {"subtask": "select materials", "dependencies": ["build a deck around your pool"]}, {"subtask": "prepare relevant tools", "dependencies": ["build a deck around your pool"]}, {"subtask": "purchase the required materials", "dependencies": ["determine the shape and size of your deck", "select materials"]}, {"subtask": "clean the site", "dependencies": ["obtain building permits", "purchase the required materials", "prepare relevant tools"]}, {"subtask": "excavate the foundation pit", "dependencies": ["clean the site"]}, {"subtask": "build the deck frame", "dependencies": ["clean the site"]}, {"subtask": "install foundation supports(such as columns)", "dependenies": ["excavate the foundation pit"]}, {"subtask": "install horizontal support beams on the frame", "dependencies": ["build the deck frame"]}, {"subtask": "lay deck planks", "dependencies": ["install foundation supports(such as columns)", "install horizontal support beams on the frame"]}, {"subtask": "install guardrail", "dependencies": ["install foundation supports(such as columns)", "install horizontal support beams on the frame"]}, {"subtask": "clean and decorate deck", "dependencies": ["lay deck planks", "install guardrail"]}]"""
user_combine_small_and_LLM_1 = """Now I give the task and the corresponding subtask, and you give the dependency relationship of the subtask.
task: (maintask)
subtasks: (all_sub_tasks)
raw_dependenies: (raw_dependencies)
Answer:"""

prompt_tree_1 = """In reality, there are many complex tasks, which can be broken down into many subtasks to solve. These subtasks often have dependencies (some subtasks must wait for other subtasks to complete before they can start). Now I give you the task and the corresponding subtasks. You need to construct them into a tree based on the dependencies between the subtasks so that some subtasks can run in parallel. You need to increase the parallelism of the subtasks as much as possible, that is, the phenomenon that more subtasks depend on the same subtask occurs more frequently. You need to return your answer in JSON format: "[{"subtask": ... "dependencies": [....]}, ......]", "subtask" represents the current subtask, and the subtasks appearing in "dependencies" need to be completed before the current subtask.

Here are some examples:
Example 1:
task: make dinner
subtasks: ["confirm menu", "wash rice", "set up the table", "prepare ingredients", "cut vegetables", "clean and slicing meat", "make soup", "prepare dishes", "prepare dishes", "cook rice", "place the plate", "place bowl and chopsticks", "serve the food"]
Answer: [{"subtask": "confirm menu", "dependencies": ["make dinner"]}, {"subtask": "wash rice", "dependencies": ["make dinner"]}, {"subtask": "set up the table", "dependencies": ["make dinner"]}, {"subtask": "prepare ingredients", "dependencies": [""confirm menu"]}, {"subtask": "cut vegetables", "dependencies": ["prepare ingredients"]}, {"subtask": "clean and slicing meat": , "dependencies": ["prepare ingredients"]}, {"subtask": "make soup", "dependencies": ["prepare ingredients"]}, {"subtask": "prepare dishes" , "dependencies": ["cut vegetables", "clean and slicing meat"]}, {"subtask": "cook rice", "dependencies": ["wash rice"]}, {"subtask": "place the plate", "dependencies": ["set up the table"]}, {"subtask": "place bowl and chopsticks", "dependencies": ["set up the table"]}, {"subtask": "serve the food", "dependencies": ["prepare dishes", "make soup", "cook rice", "place the plate", "place bowl and chopsticks"]}]

Example 2:
task: build a deck around your pool
subtasks: ["obtain building permits", "determine the shape and size of your deck", "select materials", "prepare relevant tools", "purchase the required materials", "clean the site", "excavate the foundation pit", "build the deck frame", "install foundation supports(such as columns)", "install horizontal support beams on the frame", "lay deck planks", "install guardrail", "clean and decorate deck"]
Answer: [{"subtask": "obtain building permits", "dependencies": ["build a deck around your pool"]}, {"subtask": "determine the shape and size of your deck", "dependencies": ["build a deck around your pool"]}, {"subtask": "select materials", "dependencies": ["build a deck around your pool"]}, {"subtask": "prepare relevant tools", "dependencies": ["build a deck around your pool"]}, {"subtask": "purchase the required materials", "dependencies": ["determine the shape and size of your deck", "select materials"]}, {"subtask": "clean the site", "dependencies": ["obtain building permits", "purchase the required materials", "prepare relevant tools"]}, {"subtask": "excavate the foundation pit", "dependencies": ["clean the site"]}, {"subtask": "build the deck frame", "dependencies": ["clean the site"]}, {"subtask": "install foundation supports(such as columns)", "dependenies": ["excavate the foundation pit"]}, {"subtask": "install horizontal support beams on the frame", "dependencies": ["build the deck frame"]}, {"subtask": "lay deck planks", "dependencies": ["install foundation supports(such as columns)", "install horizontal support beams on the frame"]}, {"subtask": "install guardrail", "dependencies": ["install foundation supports(such as columns)", "install horizontal support beams on the frame"]}, {"subtask": "clean and decorate deck", "dependencies": ["lay deck planks", "install guardrail"]}]"""
user_tree_1 = """Now I give the task and the corresponding subtask, and you give the dependency relationship of the subtask
task: (maintask)
subtasks: (all_sub_tasks)
Answer:"""


prompt_linear_1 = """You are a helpful assistant. In reality, there are many complex tasks, which can be broken down into many subtasks. Now I give you the task and the corresponding subtasks, but the subtasks are not sorted. You need to sort them according to the dependencies between the subtasks. The format of the answer you return is '[\"subtask1\", \"sutask2\", .....]', and you need to return the answer strictly in JSON format."""
user_linear_1 = """task: (task)
subtasks: (subtasks)"""


##################################################22222

prompt_combine_small_and_LLM_version_2 = """In reality, there are many complex tasks, which can be broken down into many subtasks to solve. These subtasks often have dependencies (some subtasks must wait for other subtasks to complete before they can start). Now I give you the task and the corresponding subtasks. You need to construct them into a tree based on the dependencies between the subtasks so that some subtasks can run in parallel. You need to increase the parallelism of the subtasks as much as possible, that is, the phenomenon that more subtasks depend on the same subtask occurs more frequently. You need to return your answer in JSON format: "[{"subtask": ... "dependencies": [....]}, ......]", "subtask" represents the current subtask, and the subtasks appearing in "dependencies" need to be completed before the current subtask.

Here are some examples:
Example 1:
task: make dinner
subtasks: ["confirm menu", "wash rice", "set up the table", "prepare ingredients", "cut vegetables", "clean and slicing meat", "make soup", "prepare dishes", "prepare dishes", "cook rice", "place the plate", "place bowl and chopsticks", "serve the food"]
Answer: [{"subtask": "confirm menu", "dependencies": ["make dinner"]}, {"subtask": "wash rice", "dependencies": ["make dinner"]}, {"subtask": "set up the table", "dependencies": ["make dinner"]}, {"subtask": "prepare ingredients", "dependencies": [""confirm menu"]}, {"subtask": "cut vegetables", "dependencies": ["prepare ingredients"]}, {"subtask": "clean and slicing meat": , "dependencies": ["prepare ingredients"]}, {"subtask": "make soup", "dependencies": ["prepare ingredients"]}, {"subtask": "prepare dishes" , "dependencies": ["cut vegetables", "clean and slicing meat"]}, {"subtask": "cook rice", "dependencies": ["wash rice"]}, {"subtask": "place the plate", "dependencies": ["set up the table"]}, {"subtask": "place bowl and chopsticks", "dependencies": ["set up the table"]}, {"subtask": "serve the food", "dependencies": ["prepare dishes", "make soup", "cook rice", "place the plate", "place bowl and chopsticks"]}]

Example 2:
task: build a deck around your pool
subtasks: ["obtain building permits", "determine the shape and size of your deck", "select materials", "prepare relevant tools", "purchase the required materials", "clean the site", "excavate the foundation pit", "build the deck frame", "install foundation supports(such as columns)", "install horizontal support beams on the frame", "lay deck planks", "install guardrail", "clean and decorate deck"]
Answer: [{"subtask": "obtain building permits", "dependencies": ["build a deck around your pool"]}, {"subtask": "determine the shape and size of your deck", "dependencies": ["build a deck around your pool"]}, {"subtask": "select materials", "dependencies": ["build a deck around your pool"]}, {"subtask": "prepare relevant tools", "dependencies": ["build a deck around your pool"]}, {"subtask": "purchase the required materials", "dependencies": ["determine the shape and size of your deck", "select materials"]}, {"subtask": "clean the site", "dependencies": ["obtain building permits", "purchase the required materials", "prepare relevant tools"]}, {"subtask": "excavate the foundation pit", "dependencies": ["clean the site"]}, {"subtask": "build the deck frame", "dependencies": ["clean the site"]}, {"subtask": "install foundation supports(such as columns)", "dependenies": ["excavate the foundation pit"]}, {"subtask": "install horizontal support beams on the frame", "dependencies": ["build the deck frame"]}, {"subtask": "lay deck planks", "dependencies": ["install foundation supports(such as columns)", "install horizontal support beams on the frame"]}, {"subtask": "install guardrail", "dependencies": ["install foundation supports(such as columns)", "install horizontal support beams on the frame"]}, {"subtask": "clean and decorate deck", "dependencies": ["lay deck planks", "install guardrail"]}]"""
user_combine_small_and_LLM_version_2 = """Now I give the task and the corresponding subtask, and you give the dependency relationship of the subtask.
task: (maintask)
subtasks: (all_sub_tasks)
Answer:"""

Observation_prompt_version_2 = """Now I will give you the derivation results of others, which may not be completely correct. As a reference, you need to draw on this reference to revise your answer.
"small_model_answer": (small_model_dependencies)
Answer:"""

##################################################22222

