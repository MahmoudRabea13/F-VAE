import gradio as gr
from functions import generate_faces


def update_slider(method):
    import gradio as gr
    if method == "noise_interp":
        return gr.update(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="α (Alpha)")
    elif method == "latent_walk":
        return gr.update(minimum=0.0, maximum=3.0, value=0.6, step=0.1, label="Walk Scale")
    elif method == "noise_ramp":
        return gr.update(minimum=0.5, maximum=5.0, value=2.0, step=0.1, label="Ramp Power")

with gr.Blocks(theme=gr.themes.Soft(primary_hue="purple", secondary_hue="gray", neutral_hue="slate"), title="VAE Facial Generator") as demo:
    gr.Markdown("# 🎭 F-VAE: Facial Variation Generator\nUpload a face image and explore realistic variations using latent space sampling.")

    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(label="Upload Image", type="pil", height=350, width=350)
            method_select = gr.Dropdown(choices=["noise_interp", "latent_walk", "noise_ramp"], value="noise_interp", label="Variation Method")
            param_slider = gr.Slider(minimum=0, maximum=1, value=0.5, step=0.01, label="Parameter")

        with gr.Column(scale=2):
            result_img = gr.Image(label="Generated Variations", type="pil", height=320)
            
            with gr.Row():
                face_only = gr.Image(label="Detected Face", type="pil", height=150, width=150)
                status = gr.Textbox(label="Status", interactive=False, scale=1)

    method_select.change(update_slider, inputs=method_select, outputs=param_slider)

    inputs = [input_img, method_select, param_slider]
    outputs = [result_img, face_only, status]

    for inp in inputs:
        inp.change(fn=generate_faces, inputs=inputs, outputs=outputs)

demo.launch()
