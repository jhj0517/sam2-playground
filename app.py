import gradio as gr
import os

from modules.sam_inference import SamInference
from modules.model_downloader import DEFAULT_MODEL_TYPE
from modules.paths import OUTPUT_DIR
from modules.utils import open_folder

sam_inf = SamInference()

with gr.Blocks() as app:
    with gr.Row():
        with gr.Column(scale=5):
            img_input = gr.Image(label="Input image here")
        with gr.Column(scale=5):
            dd_models = gr.Dropdown(label="Model", value=DEFAULT_MODEL_TYPE, choices=sam_inf.available_models)
            nb_points_per_side = gr.Number(label="points_per_side ", value=64)
            nb_points_per_batch = gr.Number(label="points_per_batch ", value=128)
            sld_pred_iou_thresh = gr.Slider(label="pred_iou_thresh ", value=0.7, minimum=0, maximum=1)
            sld_stability_score_thresh = gr.Slider(label="stability_score_thresh ", value=0.92, minimum=0,
                                                   maximum=1)
            sld_stability_score_offset = gr.Slider(label="stability_score_offset ", value=0.7, minimum=0,
                                                   maximum=1)
            nb_crop_n_layers = gr.Number(label="crop_n_layers ", value=1)
            sld_box_nms_thresh = gr.Slider(label="box_nms_thresh ", value=0.7, minimum=0,
                                           maximum=1)
            nb_crop_n_points_downscale_factor = gr.Number(label="crop_n_points_downscale_factor ", value=2)
            nb_min_mask_region_area = gr.Number(label="min_mask_region_area ", value=25)
            cb_use_m2m = gr.Checkbox(label="use_m2m ", value=True)

    with gr.Row():
        btn_generate = gr.Button("GENERATE", variant="primary")
    with gr.Row():
        gallery_output = gr.Gallery(label="Output images will be shown here")
        with gr.Column():
            output_file = gr.File(label="Generated psd file", scale=8)
            btn_open_folder = gr.Button("📁\nOpen PSD folder", scale=2)

    params = [nb_points_per_side, nb_points_per_batch, sld_pred_iou_thresh, sld_stability_score_thresh,
              sld_stability_score_offset,
              nb_crop_n_layers, sld_box_nms_thresh, nb_crop_n_points_downscale_factor, nb_min_mask_region_area,
              cb_use_m2m]
    btn_generate.click(fn=sam_inf.generate_mask_app,
                       inputs=[img_input, dd_models] + params, outputs=[gallery_output, output_file])
    btn_open_folder.click(fn=lambda: open_folder(os.path.join(OUTPUT_DIR)),
                          inputs=None, outputs=None)

app.queue().launch(inbrowser=True)