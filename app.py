import gradio as gr
from gradio_image_prompter import ImagePrompter
from typing import List, Dict, Optional, Union
import os
import yaml

from modules.sam_inference import SamInference
from modules.model_downloader import DEFAULT_MODEL_TYPE
from modules.paths import (OUTPUT_DIR, SAM2_CONFIGS_DIR)
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
        default_param_config_path = os.path.join(SAM2_CONFIGS_DIR, "default_hparams.yaml")
        with open(default_param_config_path, 'r') as file:
            self.hparams = yaml.safe_load(file)

    @staticmethod
    def on_mode_change(mode: str):
        return [
            gr.Image(visible=mode == AUTOMATIC_MODE),
            ImagePrompter(visible=mode == BOX_PROMPT_MODE),
            gr.Accordion(visible=mode == AUTOMATIC_MODE),
        ]

    def mask_parameters(self,
                        hparams: Optional[Dict] = None):
        if hparams is None:
            hparams = self.hparams["mask_hparams"]
        mask_components = [
            gr.Number(label="points_per_side ", value=hparams["points_per_side"], interactive=True),
            gr.Number(label="points_per_batch ", value=hparams["points_per_batch"], interactive=True),
            gr.Slider(label="pred_iou_thresh ", value=hparams["pred_iou_thresh"], minimum=0, maximum=1,
                      interactive=True),
            gr.Slider(label="stability_score_thresh ", value=hparams["stability_score_thresh"], minimum=0,
                      maximum=1, interactive=True),
            gr.Slider(label="stability_score_offset ", value=hparams["stability_score_offset"], minimum=0,
                      maximum=1),
            gr.Number(label="crop_n_layers ", value=hparams["crop_n_layers"]),
            gr.Slider(label="box_nms_thresh ", value=hparams["box_nms_thresh"], minimum=0, maximum=1),
            gr.Number(label="crop_n_points_downscale_factor ", value=hparams["crop_n_points_downscale_factor"]),
            gr.Number(label="min_mask_region_area ", value=hparams["min_mask_region_area"]),
            gr.Checkbox(label="use_m2m ", value=hparams["use_m2m"])
        ]
        return mask_components

    def launch(self):
        _mask_hparams = self.hparams["mask_hparams"]

        with self.app:
            with gr.Row():
                with gr.Column(scale=5):
                    img_input = gr.Image(label="Input image here")
                    img_input_prompter = ImagePrompter(label="Prompt image with Box & Point", type='pil',
                                                       visible=self.default_mode == BOX_PROMPT_MODE)

                with gr.Column(scale=5):
                    dd_input_modes = gr.Dropdown(label="Image Input Mode", value=self.default_mode,
                                                 choices=self.image_modes)
                    dd_models = gr.Dropdown(label="Model", value=DEFAULT_MODEL_TYPE,
                                            choices=self.sam_inf.available_models)

                    with gr.Accordion("Mask Parameters", open=False) as acc_mask_hparams:
                        mask_hparams_component = self.mask_parameters(_mask_hparams)

                    cb_multimask_output = gr.Checkbox(label="multimask_output", value=_mask_hparams["multimask_output"])

            with gr.Row():
                btn_generate = gr.Button("GENERATE", variant="primary")
            with gr.Row():
                gallery_output = gr.Gallery(label="Output images will be shown here")
                with gr.Column():
                    output_file = gr.File(label="Generated psd file", scale=9)
                    btn_open_folder = gr.Button("📁\nOpen PSD folder", scale=1)

            sources = [img_input, img_input_prompter, dd_input_modes]
            model_params = [dd_models]
            mask_hparams = mask_hparams_component + [cb_multimask_output]
            input_params = sources + model_params + mask_hparams

            btn_generate.click(fn=self.sam_inf.divide_layer,
                               inputs=input_params, outputs=[gallery_output, output_file])
            btn_open_folder.click(fn=lambda: open_folder(os.path.join(OUTPUT_DIR)),
                                  inputs=None, outputs=None)
            dd_input_modes.change(fn=self.on_mode_change,
                                  inputs=[dd_input_modes],
                                  outputs=[img_input, img_input_prompter, acc_mask_hparams])

        self.app.queue().launch(inbrowser=True)


if __name__ == "__main__":
    app = App()
    app.launch()
