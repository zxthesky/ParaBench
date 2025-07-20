import json
import re
import os
from openai import OpenAI


detect_prompt_whether_pass = """You are an outstanding task planning expert. In reality, there are many complex tasks, and you are good at breaking them down into subtasks and then solving them. There are certain dependencies between these subtasks, and some tasks need to be completed after some other tasks. Now I give you the main task and the corresponding subtask sequence. Its format is "[subtask1, subtask2, ...]". Now I execute these subtasks in order, and you need to help me judge whether they can be completed successfully. Remember that some of the subtasks have no dependencies, and there is no need to require the order between them. Now you need to tell me whether it can be completed. If it can be completed, return "[YES]". If you are not sure, return "UNKNOWN". If it cannot be completed, return "NO".
Below I give you the main task and the corresponding decomposed subtask sequence. Among them, "main_task" represents the main task, and "subtask" represents the decomposed subtask sequence."""

user_whether_pass = """
"main_task": [main_task]
"subtasks": [subtasks]"""


detect_prompt_scores = """You are an outstanding task planning expert. You are good at decomposing a task into multiple subtasks and completing tasks in parallel as much as possible according to the dependencies between tasks. Now I will give you a task for which I have made a preliminary task planning. The planning format I give is [{"subtask": subtask_1, "dependencies": [subtask_x, subtask_y, ...]}, {"subtask": subtask_2, "dependencies": [subtask_i, subtask_j, ...]}, ...]. Among them, "subtask" represents the current subtask, and "dependencies" represents the subtasks that the current "subtask" needs to depend on, that is, the current "subtask" needs to be completed after the subtasks in "dependencies". If there is no element in "dependencies", it means that the current subtask is the top-level subtask and does not need to depend on other subtasks. We hope that tasks can be completed in parallel as much as possible, that is, I hope that a subtask can appear in the "dependencies" of each "subtask" as much as possible.
Next, I will give you the main task and the corresponding task plan. "main_task" represents the main task, and "all_dependencies" represents my plan ("all_dependencies" only sorts the subtasks and does not involve "main_task"). Now you need to understand the specific "main_task" and combine it to score my task plan. The score is an integer from 0 to 10. The higher the score, the better the plan. If the score is less than 5, it means that the current answer is not very good. The return format of your answer is "{"score": your_score}"."""

user_scores = """
"main_task": [main_task]
"all_dependencies": [all_dependencies]"""


def get_response(main_task, candidate_data: list, model_type: str="Qwen", eval_model:str = "pass"):
    """
    Args:
        model_type(str): chatgpt, chatglm, Qwen ...
        candidate_data(list):  candidate_data: for example subtasks or all_dependencies
        model_type(str): Qwen or chatgpt or ...
        eval_model(str): 'pass' or 'scores'
    """
    system_content = ""
    user_content = ""
    if eval_model == "pass":
        user_content = user_whether_pass.replace("[main_task]", main_task)
        user_content = user_whether_pass.replace("[subtasks]", str(candidate_data))
        system_content = detect_prompt_whether_pass
    elif eval_model == "scores":
        user_content = user_scores.replace("[main_task]", main_task)
        user_content = user_scores.replace("[all_dependencies]", str(candidate_data))
        system_content = detect_prompt_scores


    if "chatgpt" in model_type:
        pass
    elif "Qwen" in model_type:
        now_model_type = "qwen-max-latest"
        messages=[
            {'role': 'system', 'content': system_content},
            {'role': 'user', 'content': user_content}
        ]
        client = OpenAI(
            api_key="sk-a7cfbcde223b41c2b69dcced862e3ab2",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        completion = client.chat.completions.create(
            model=now_model_type,
            messages=messages
        )
        raw_response = completion.choices[0].message.content
        return raw_response





def change_all_dependencies(raw_response):
    response = re.findall(r"\[.*\{.*\}.*\]", raw_response, re.DOTALL)[0]
    raw_model_data = json.loads(response)

    


def use_model_generate_data(raw_response):
    response = re.findall(r"\[.*\{.*\}.*\]", raw_response, re.DOTALL)[0]
    raw_model_data = json.loads(response)
    



if __name__ == '__main__':
    pass



