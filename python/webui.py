from pathlib import Path
import gradio as gr
import chatglm_cpp

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models/chatglm3-ggml-q4_0.bin"
print('####模型地址',DEFAULT_MODEL_PATH)

num_threads=8
print("####线程数",num_threads)

pipeline = chatglm_cpp.Pipeline(DEFAULT_MODEL_PATH)
res=pipeline.chat(["hi"])
print(res)

mode='chat'


def predict(input, chatbot, max_length, top_p, temperature, history):
    chatbot.append((input, ""))
    response = ""
    history.append(input)

    generation_kwargs = dict(
        max_length=max_length,
        do_sample=temperature > 0,
        top_p=top_p,
        temperature=temperature,
        num_threads=num_threads,
        stream=True,
    )
    generator = (
        pipeline.chat(history, **generation_kwargs)
        if mode == "chat"
        else pipeline.generate(input, **generation_kwargs)
    )
    for response_piece in generator:
        response += response_piece
        chatbot[-1] = (chatbot[-1][0], response)

        yield chatbot, history

    history.append(response)
    yield chatbot, history

def reset_user_input():
    return gr.update(value="")


def reset_state():
    return [], []

with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">ChatGLM.cpp</h1>""")

    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=4):
            user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=8)
            submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            max_length = gr.Slider(0, 2048, value=2048, step=1.0, label="Maximum Length", interactive=True)
            top_p = gr.Slider(0, 1, value=0.7, step=0.01, label="Top P", interactive=True)
            temperature = gr.Slider(0, 1, value=0.9, step=0.01, label="Temperature", interactive=True)
            emptyBtn = gr.Button("Clear History")

    history = gr.State([])

    submitBtn.click(
        predict, [user_input, chatbot, max_length, top_p, temperature, history], [chatbot, history], show_progress=True
    )
    submitBtn.click(reset_user_input, [], [user_input])

    emptyBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)


def start():
    demo.queue().launch(share=False, inbrowser=True)

