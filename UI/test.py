import gradio as gr


def show(image):
    return image


with gr.Blocks() as demo:
    # 标题的位置
    gr.Markdown(
        "<div align='center'><font face='华文行楷'><font size='70'>基于深度学习的细粒度图像检索系统</font></div>"
    )

    with gr.Column():
        with gr.Row():
            Image_input = gr.Image(label="原图")
            # a = gr.inputs.File()
            Result = gr.Image(label="处理后", interactive=False)
        with gr.Row():
            submit_btn = gr.Button("Calculate")
            clear_btn = gr.ClearButton()
            clear_btn.add([Image_input, Result])

        submit_btn.click(fn=show, inputs=[Image_input], outputs=[Result])
        clear_btn.click(fn=None)

if __name__ == '__main__':
    demo.launch(auth=("admin", "123456"))
