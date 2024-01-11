from llama_cpp import Llama

class LlamaAssistant:
    # chat_format = functionary llama-2
    def __init__(self, model_path,chat_format="llama-2",n_ctx=0,embedding=True):
        self.llm = Llama(model_path=model_path, chat_format=chat_format,n_ctx=n_ctx,embedding=embedding)
        self.chat=self.llm.create_chat_completion
        # self.llm.n_ctx()
        #  create_embedding // embedding=True
        # reset
    def pre_messages(self, user_prompt):
        system_prompt = {
            "role": "system",
            "content": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. The assistant calls functions with appropriate input when necessary"
        }
        
        return [
            system_prompt,
            user_prompt
        ]
    
    def run(self, user_prompt, tools, tool_choice):
        user_prompt = {
            "role": "user",
            "content": user_prompt
        }
 
        messages = self.pre_messages(user_prompt)
        # print(messages)
        return self.llm.create_chat_completion(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice
        )
    
    def embedding(self,texts):
        
        result=[]
        for data in self.llm.create_embedding(texts)["data"]:
            result.append(data["embedding"])

        print(len(result))
        # list(map(float, self.llm.create_embedding(input)["data"][0]["embedding"]))
        return result

    
    

# 示例用法
# assistant = LlamaAssistant(
#     model_path="models/llama/functionary-7b-v1.Q5_K.gguf",
#     chat_format="functionary"
#     )
# user_input = "Extract Jason is 25 years old"
# tools = [{
#     "type": "function",
#     "function": {
#         "name": "UserDetail",
#         "parameters": {
#             "type": "object",
#             "title": "UserDetail",
#             "properties": {
#                 "name": {
#                     "title": "Name",
#                     "type": "string"
#                 },
#                 "age": {
#                     "title": "Age",
#                     "type": "integer"
#                 }
#             },
#             "required": ["name", "age"]
#         }
#     }
# }]
# tool_choice = {
#     "type": "function",
#     "function": {
#         "name": "UserDetail"
#     }
# }

# result=assistant.run(user_input, tools, tool_choice)
# # print('#result#',result)
# # result=assistant.embedding([user_input,user_input] )
# print('#result#',result)


