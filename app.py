import gradio as gr
from gradio_image_prompter import ImagePrompter
import os

from modules.sam_inference import SamInference
from modules.model_downloader import DEFAULT_MODEL_TYPE
from modules.paths import OUTPUT_DIR
from modules.utils import open_folder
from modules.constants import (AUTOMATIC_MODE, BOX_PROMPT_MODE)


class App:
    def __init__(self,
                 args=None):
        self.app = gr.Blocks()
        self.args = args
        self.sam_inf = SamInference()
        self.image_modes = [AUTOMATIC_MODE, BOX_PROMPT_MODE]
        self.default_mode = AUTOMATIC_MODE

    @staticmethod
    def on_mode_change(mode: str):
        return [
            gr.Image(visible=mode == AUTOMATIC_MODE),
            ImagePrompter(visible=mode == BOX_PROMPT_MODE),
            gr.Accordion(visible=mode == AUTOMATIC_MODE)
        ]

    def launch(self):
        with self.app:
            with gr.Row():
                with gr.Column(scale=5):
                    img_input = gr.Image(label="Input image here")
                    img_input_prompter = ImagePrompter(label="Prompt image with Box & Point",
                                                       visible=self.default_mode == BOX_PROMPT_MODE)

                with gr.Column(scale=5):
                    dd_input_modes = gr.Dropdown(label="Image Input Mode", value=self.default_mode,
                                                 choices=self.image_modes)
                    dd_models = gr.Dropdown(label="Model", value=DEFAULT_MODEL_TYPE,
                                            choices=self.sam_inf.available_models)

                    with gr.Accordion("Mask Parameters", open=False) as mask_hparams:
                        nb_points_per_side = gr.Number(label="points_per_side ", value=64, interactive=True)
                        nb_points_per_batch = gr.Number(label="points_per_batch ", value=128, interactive=True)
                        sld_pred_iou_thresh = gr.Slider(label="pred_iou_thresh ", value=0.7, minimum=0, maximum=1,
                                                        interactive=True)
                        sld_stability_score_thresh = gr.Slider(label="stability_score_thresh ", value=0.92, minimum=0,
                                                               maximum=1, interactive=True)
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
                    output_file = gr.File(label="Generated psd file", scale=9)
                    btn_open_folder = gr.Button("📁\nOpen PSD folder", scale=1)

            sources = [img_input]
            model_params = [dd_models]
            auto_mask_hparams = [nb_points_per_side, nb_points_per_batch, sld_pred_iou_thresh,
                                 sld_stability_score_thresh, sld_stability_score_offset, nb_crop_n_layers,
                                 sld_box_nms_thresh, nb_crop_n_points_downscale_factor, nb_min_mask_region_area,
                                 cb_use_m2m]

            btn_generate.click(fn=self.sam_inf.generate_mask_app,
                               inputs=sources + model_params + auto_mask_hparams, outputs=[gallery_output, output_file])
            btn_open_folder.click(fn=lambda: open_folder(os.path.join(OUTPUT_DIR)),
                                  inputs=None, outputs=None)

            dd_input_modes.change(fn=self.on_mode_change,
                                  inputs=[dd_input_modes],
                                  outputs=[img_input, img_input_prompter, mask_hparams])

        self.app.queue().launch(inbrowser=True)


if __name__ == "__main__":
    app = App()
    app.launch()
