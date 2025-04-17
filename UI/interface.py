import gradio as gr
import retrival_function as retrival
import torch


def calculate_feature_code(image):
    return image.shape()

def a(num):
    return num

def clear_cache():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def module_visible():
    return gr.update(visible=True)


def main():
    with gr.Blocks() as demo:
        # gradio界面标题设计
        gr.Markdown("<div align='center'><font face='华文行楷'><font size='70'>基于深度学习的细粒度图像检索系统</font></div>")
        # gr.Markdown("# 基于深度学习的细粒度图像检索系统")
        feature_code = gr.State()
        feature_code_db = gr.State()
        kind = gr.State()

        # 图像检索分页界面
        with gr.Tab("Image Retrival"):
            with gr.Column():
                with gr.Row():
                    Input_image = gr.Image(label="Upload Image")
                    with gr.Column():
                        # with gr.Row():
                        Category = gr.Text(label="Category")
                        Type = gr.Text(label="Type", min_width=50)
                        feature_output = gr.Text(label="Image Feature Code", lines=5, max_lines=5, min_width=50)

                similarity_num = gr.Slider(minimum=5, maximum=15, step=1, label="The Number Of Retrival Return Number")

                with gr.Row():
                    calculate_button = gr.Button("Submit")
                    retrival_button = gr.Button("Retrival")
                    clear_btn = gr.ClearButton()

                calculate_button.click(fn=retrival.calculate_feature_code, inputs=Input_image,
                                       outputs=[feature_code, feature_output, kind, Category, Type]
                                       )

            # 设置检索结果摆放的组件，gradio的组件无法在运行的时候进行添加
            result = gr.Markdown("<div align='center'><font face='华文行楷'><font size='50'>检索结果</font></div>", visible=False)
            with gr.Column():
                with gr.Row():
                    Image_1 = gr.Image(visible=False, label="NO.1")
                    with gr.Column():
                        Similar_1 = gr.Text(label="Similarity", visible=False)
                        Category_1 = gr.Text(label="Image Type", visible=False)
                with gr.Row():
                    Image_2 = gr.Image(visible=False, label="NO.2")
                    with gr.Column():
                        Similar_2 = gr.Text(label="Similarity", visible=False)
                        Category_2 = gr.Text(label="Image Type", visible=False)
                with gr.Row():
                    Image_3 = gr.Image(visible=False, label="NO.3")
                    with gr.Column():
                        Similar_3 = gr.Text(label="Similarity", visible=False)
                        Category_3 = gr.Text(label="Image Type", visible=False)
                with gr.Row():
                    Image_4 = gr.Image(visible=False, label="NO.4")
                    with gr.Column():
                        Similar_4 = gr.Text(label="Similarity", visible=False)
                        Category_4 = gr.Text(label="Image Type", visible=False)
                with gr.Row():
                    Image_5 = gr.Image(visible=False, label="NO.5")
                    with gr.Column():
                        Similar_5 = gr.Text(label="Similarity", visible=False)
                        Category_5 = gr.Text(label="Image Type", visible=False)
                with gr.Row():
                    Image_6 = gr.Image(visible=False, label="NO.6")
                    with gr.Column():
                        Similar_6 = gr.Text(label="Similarity", visible=False)
                        Category_6 = gr.Text(label="Image Type", visible=False)
                with gr.Row():
                    Image_7 = gr.Image(visible=False, label="NO.7")
                    with gr.Column():
                        Similar_7 = gr.Text(label="Similarity", visible=False)
                        Category_7 = gr.Text(label="Image Type", visible=False)
                with gr.Row():
                    Image_8 = gr.Image(visible=False, label="NO.8")
                    with gr.Column():
                        Similar_8 = gr.Text(label="Similarity", visible=False)
                        Category_8 = gr.Text(label="Image Type", visible=False)
                with gr.Row():
                    Image_9 = gr.Image(visible=False, label="NO.9")
                    with gr.Column():
                        Similar_9 = gr.Text(label="Similarity", visible=False)
                        Category_9 = gr.Text(label="Image Type", visible=False)
                with gr.Row():
                    Image_10 = gr.Image(visible=False, label="NO.10")
                    with gr.Column():
                        Similar_10 = gr.Text(label="Similarity", visible=False)
                        Category_10 = gr.Text(label="Image Type", visible=False)
                with gr.Row():
                    Image_11 = gr.Image(visible=False, label="NO.11")
                    with gr.Column():
                        Similar_11 = gr.Text(label="Similarity", visible=False)
                        Category_11 = gr.Text(label="Image Type", visible=False)
                with gr.Row():
                    Image_12 = gr.Image(visible=False, label="NO.12")
                    with gr.Column():
                        Similar_12 = gr.Text(label="Similarity", visible=False)
                        Category_12 = gr.Text(label="Image Type", visible=False)
                with gr.Row():
                    Image_13 = gr.Image(visible=False, label="NO.13")
                    with gr.Column():
                        Similar_13 = gr.Text(label="Similarity", visible=False)
                        Category_13 = gr.Text(label="Image Type", visible=False)
                with gr.Row():
                    Image_14 = gr.Image(visible=False, label="NO.14")
                    with gr.Column():
                        Similar_14 = gr.Text(label="Similarity", visible=False)
                        Category_14 = gr.Text(label="Image Type", visible=False)
                with gr.Row():
                    Image_15 = gr.Image(visible=False, label="NO.15")
                    with gr.Column():
                        Similar_15 = gr.Text(label="Similarity", visible=False)
                        Category_15 = gr.Text(label="Image Type", visible=False)

            retrival_button.click(fn=retrival.retrival_show_message, inputs=[similarity_num, feature_code, kind],
                                  outputs=[Image_1, Image_1, Category_1, Category_1, Similar_1, Similar_1,
                                           Image_2, Image_2, Category_2, Category_2, Similar_2, Similar_2,
                                           Image_3, Image_3, Category_3, Category_3, Similar_3, Similar_3,
                                           Image_4, Image_4, Category_4, Category_4, Similar_4, Similar_4,
                                           Image_5, Image_5, Category_5, Category_5, Similar_5, Similar_5,
                                           Image_6, Image_6, Category_6, Category_6, Similar_6, Similar_6,
                                           Image_7, Image_7, Category_7, Category_7, Similar_7, Similar_7,
                                           Image_8, Image_8, Category_8, Category_8, Similar_8, Similar_8,
                                           Image_9, Image_9, Category_9, Category_9, Similar_9, Similar_9,
                                           Image_10, Image_10, Category_10, Category_10, Similar_10, Similar_10,
                                           Image_11, Image_11, Category_11, Category_11, Similar_11, Similar_11,
                                           Image_12, Image_12, Category_12, Category_12, Similar_12, Similar_12,
                                           Image_13, Image_13, Category_13, Category_13, Similar_13, Similar_13,
                                           Image_14, Image_14, Category_14, Category_14, Similar_14, Similar_14,
                                           Image_15, Image_15, Category_15, Category_15, Similar_15, Similar_15, result]
                                  )

            clear_btn.add([Input_image, Category, Type, feature_output,
                           Image_1, Image_1, Category_1, Category_1, Similar_1, Similar_1,
                           Image_2, Image_2, Category_2, Category_2, Similar_2, Similar_2,
                           Image_3, Image_3, Category_3, Category_3, Similar_3, Similar_3,
                           Image_4, Image_4, Category_4, Category_4, Similar_4, Similar_4,
                           Image_5, Image_5, Category_5, Category_5, Similar_5, Similar_5,
                           Image_6, Image_6, Category_6, Category_6, Similar_6, Similar_6,
                           Image_7, Image_7, Category_7, Category_7, Similar_7, Similar_7,
                           Image_8, Image_8, Category_8, Category_8, Similar_8, Similar_8,
                           Image_9, Image_9, Category_9, Category_9, Similar_9, Similar_9,
                           Image_10, Image_10, Category_10, Category_10, Similar_10, Similar_10,
                           Image_11, Image_11, Category_11, Category_11, Similar_11, Similar_11,
                           Image_12, Image_12, Category_12, Category_12, Similar_12, Similar_12,
                           Image_13, Image_13, Category_13, Category_13, Similar_13, Similar_13,
                           Image_14, Image_14, Category_14, Category_14, Similar_14, Similar_14,
                           Image_15, Image_15, Category_15, Category_15, Similar_15, Similar_15, result
                           ])
            clear_btn.click(fn=None)

        # 数据库管理分页界面
        with gr.Tab("Database Management"):
            with gr.Column():
                with gr.Row():
                    Insert_image = gr.Image(label="Upload Image")
                    with gr.Column():
                        Image_path = gr.Text(label="Please Input Image Absolutely Path")
                        Insert_category = gr.Text(label="Category")
                        Insert_type = gr.Text(label="Type")
                        Insert_feature = gr.Text(label="Image Feature Code", max_lines=5, lines=5)
                with gr.Row():
                    get_msg_button = gr.Button("Calculate")
                    insert_button = gr.Button("Insert")

            get_msg_button.click(fn=retrival.calculate_feature_code_db, inputs=[Insert_image, Image_path],
                                 outputs=[feature_code_db, Insert_feature, Insert_category, Insert_type, Image_path])

    demo.launch(auth=("admin", "113320"))
    # demo.launch()


if __name__ == '__main__':
    main()
