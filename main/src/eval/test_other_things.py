# from openai import OpenAI

# client = OpenAI(api_key="sk-ybkpixhhmwdzcuicxgjfmmsjlttdcpspupubrcrnrojovjfr", base_url="https://api.siliconflow.cn/v1")
# response = client.chat.completions.create(
#     model='meta-llama/Llama-3.3-70B-Instruct',
#     messages=[
#         {'role': 'user', 
#         'content': "中国大模型行业2025年将会迎来哪些机遇和挑战"}
#     ]
# )
# print(response)
# print("----------------------------")
# print("----------------------------")
# print("----------------------------")
# print(response.choices)
# print("----------------------------")
# print(response.choices[0].message.content)
# # for chunk in response:
# #     print(chunk.choices[0].delta.content, end='')


a = [1,2,3,4]

print(a[::-1])
