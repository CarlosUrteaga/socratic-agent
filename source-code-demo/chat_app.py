import gradio as gr
from socratic_agent import SocraticController

ctrl = SocraticController(tau=0.7, offline=False)

SYSTEM_HINT = (
    "Tips:\n"
    "• Start a RAG lesson with: "
    "RAG[https://en.wikipedia.org/wiki/Retrieval-augmented_generation,https://fastapi.tiangolo.com/]: Explain RAG...\n"
    "• Or ask numeric questions to see ASK→PROBE→SUMMARIZE→VERIFY.\n"
)

def respond(user_msg, history):
    out = ctrl.step(user_msg) or {}
    text = out.get("text","")
    act = out.get("act"); stance = out.get("stance")
    header = f"[{act}/{stance}] "
    return history + [[user_msg, header + text]]

with gr.Blocks(theme="soft") as demo:
    gr.Markdown("# Socratic Agent (with RAG)")
    gr.Markdown(SYSTEM_HINT)
    chat = gr.Chatbot(height=480, show_label=False)
    msg = gr.Textbox(placeholder="Type here…", autofocus=True)
    send = gr.Button("Send")

    def _send(m, h): 
        new_h = respond(m, h)
        return "", new_h

    msg.submit(_send, [msg, chat], [msg, chat])
    send.click(_send, [msg, chat], [msg, chat])

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
