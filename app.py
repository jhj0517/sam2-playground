import gradio as gr
from gradio_image_prompter import ImagePrompter
from gradio_image_prompter.image_prompter import PromptData
from typing import List, Dict, Optional, Union
import os
import yaml

from modules.sam_inference import SamInference
from modules.model_downloader import DEFAULT_MODEL_TYPE
from modules.paths import (OUTPUT_DIR, OUTPUT_PSD_DIR, SAM2_CONFIGS_DIR, TEMP_DIR)
from modules.utils import open_folder
from modules.constants import (AUTOMATIC_MODE, BOX_PROMPT_MODE)
from modules.video_utils import extract_frames, get_frames_from_dir


class App:
    def __init__(self,
                 args=None):
        self.demo = gr.Blocks()
        self.args = args
        self.sam_inf = SamInference()
        self.image_modes = [AUTOMATIC_MODE, BOX_PROMPT_MODE]
        self.default_mode = BOX_PROMPT_MODE
        default_param_config_path = os.path.join(SAM2_CONFIGS_DIR, "default_hparams.yaml")
        with open(default_param_config_path, 'r') as file:
            self.hparams = yaml.safe_load(file)

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

    @staticmethod
    def on_mode_change(mode: str):
        return [
            gr.Image(visible=mode == AUTOMATIC_MODE),
            ImagePrompter(visible=mode == BOX_PROMPT_MODE),
            gr.Accordion(visible=mode == AUTOMATIC_MODE),
        ]

    def on_video_upload(self, vid_input: str):
        output_temp_dir = TEMP_DIR
        extract_frames(vid_input=vid_input, output_temp_dir=output_temp_dir)
        frames = get_frames_from_dir(vid_dir=output_temp_dir)
        # self.sam_inf.init_video_inference_state(output_temp_dir)
        return [
            ImagePrompter(label="Prompt image with Box & Point", value=frames[0]),
            gr.Slider(label="Frame Indexes", value=0, interactive=True, step=1, minimum=0, maximum=(len(frames)-1))
        ]

    @staticmethod
    def on_frame_change(frame_idx: int):
        temp_dir = TEMP_DIR
        frames = get_frames_from_dir(vid_dir=temp_dir)
        selected_frame = frames[frame_idx]
        return ImagePrompter(elem_id="vid-prompter-index", label=f"Prompt image with Box & Point #{frame_idx}",
                             value=selected_frame)

    @staticmethod
    def on_prompt_change(prompt: Dict):
        image, points = prompt["image"], prompt["points"]
        return gr.Image(label="Preview", value=image)

    def launch(self):
        _mask_hparams = self.hparams["mask_hparams"]

        with self.demo:
            with gr.Tabs():
                with gr.TabItem("Layer Divider"):
                    with gr.Row():
                        with gr.Column(scale=5):
                            img_input = gr.Image(label="Input image here", visible=self.default_mode == AUTOMATIC_MODE)
                            img_input_prompter = ImagePrompter(label="Prompt image with Box & Point", type='pil',
                                                               visible=self.default_mode == BOX_PROMPT_MODE)

                        with gr.Column(scale=5):
                            dd_input_modes = gr.Dropdown(label="Image Input Mode", value=self.default_mode,
                                                         choices=self.image_modes)
                            dd_models = gr.Dropdown(label="Model", value=DEFAULT_MODEL_TYPE,
                                                    choices=self.sam_inf.available_models)

                            with gr.Accordion("Mask Parameters", open=False, visible=self.default_mode == AUTOMATIC_MODE) as acc_mask_hparams:
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
                    btn_open_folder.click(fn=lambda: open_folder(os.path.join(OUTPUT_PSD_DIR)),
                                          inputs=None, outputs=None)
                    dd_input_modes.change(fn=self.on_mode_change,
                                          inputs=[dd_input_modes],
                                          outputs=[img_input, img_input_prompter, acc_mask_hparams])

                with gr.TabItem("Mosaic Filter"):
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=2):
                            vid_input = gr.Video(label="Input Video here", scale=3)
                        with gr.Column(scale=8):
                            with gr.Row():
                                vid_frame_prompter = ImagePrompter(elem_id="vid-prompter",
                                                                   label="Prompt image with Box & Point  ",
                                                                   interactive=True, scale=5)
                                img_preview = gr.Image(label="Preview", interactive=False, scale=5)
                            sld_frame_selector = gr.Slider(label="Frame Index", interactive=False)

                    with gr.Row():
                        btn_generate = gr.Button("GENERATE", variant="primary")
                    with gr.Row():
                        gallery_output = gr.Gallery(label="Output images will be shown here")
                        with gr.Column():
                            output_file = gr.File(label="Generated psd file", scale=9)
                            btn_open_folder = gr.Button("📁\nOpen PSD folder", scale=1)

                    vid_input.change(fn=self.on_video_upload,
                                     inputs=[vid_input],
                                     outputs=[vid_frame_prompter, sld_frame_selector])
                    sld_frame_selector.change(fn=self.on_frame_change,
                                              inputs=[sld_frame_selector],
                                              outputs=[vid_frame_prompter],)
                    vid_frame_prompter.change(fn=self.on_prompt_change,
                                              inputs=[vid_frame_prompter],
                                              outputs=[img_preview])

        self.demo.queue().launch(inbrowser=True)


if __name__ == "__main__":
    demo = App()
    demo.launch()
