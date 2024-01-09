from llama_cpp import Llama

class LlamaAssistant:
    def __init__(self, model_path):
        self.llm = Llama(model_path=model_path, chat_format="functionary")
    
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
        
        self.llm.create_chat_completion(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice
        )

# 示例用法
assistant = LlamaAssistant(model_path="path/to/model")
user_input = "Extract Jason is 25 years old"
tools = [{
    "type": "function",
    "function": {
        "name": "UserDetail",
        "parameters": {
            "type": "object",
            "title": "UserDetail",
            "properties": {
                "name": {
                    "title": "Name",
                    "type": "string"
                },
                "age": {
                    "title": "Age",
                    "type": "integer"
                }
            },
            "required": ["name", "age"]
        }
    }
}]
tool_choice = [{
    "type": "function",
    "function": {
        "name": "UserDetail"
    }
}]

assistant.run(user_input, tools, tool_choice)

