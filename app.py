import argparse
import gradio as gr
from gradio_image_prompter import ImagePrompter
from typing import List, Dict, Optional, Union
import os
import yaml

from modules.logger_util import get_logger
from modules.html_constants import (HEADER, DEFAULT_THEME, CSS)
from modules.sam_inference import SamInference
from modules.model_downloader import DEFAULT_MODEL_TYPE
from modules.paths import (OUTPUT_DIR, OUTPUT_PSD_DIR, SAM2_CONFIGS_DIR, TEMP_DIR, OUTPUT_FILTER_DIR, MODELS_DIR)
from modules.utils import open_folder
from modules.constants import (AUTOMATIC_MODE, BOX_PROMPT_MODE, PIXELIZE_FILTER, COLOR_FILTER, DEFAULT_COLOR,
                               DEFAULT_PIXEL_SIZE, SOUND_FILE_EXT, IMAGE_FILE_EXT, VIDEO_FILE_EXT)
from modules.video_utils import get_frames_from_dir


logger = get_logger()


class App:
    def __init__(self,
                 args: argparse.Namespace):
        self.args = args
        self.demo = gr.Blocks(
            theme=self.args.theme,
            css=CSS
        )
        self.sam_inf = SamInference(
            model_dir=self.args.model_dir,
            output_dir=self.args.output_dir
        )
        logger.info(f'device "{self.sam_inf.device}" is detected')
        self.image_modes = [AUTOMATIC_MODE, BOX_PROMPT_MODE]
        self.default_mode = BOX_PROMPT_MODE
        self.filter_modes = [PIXELIZE_FILTER, COLOR_FILTER]
        self.default_filter = PIXELIZE_FILTER
        self.default_color = DEFAULT_COLOR
        self.default_pixel_size = DEFAULT_PIXEL_SIZE
        default_hparam_config_path = os.path.join(SAM2_CONFIGS_DIR, "default_hparams.yaml")
        with open(default_hparam_config_path, 'r') as file:
            self.default_hparams = yaml.safe_load(file)

    def mask_parameters(self,
                        hparams: Optional[Dict] = None):
        if hparams is None:
            hparams = self.default_hparams["mask_hparams"]
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

    @staticmethod
    def on_filter_mode_change(mode: str):
        return [
            gr.ColorPicker(visible=mode == COLOR_FILTER),
            gr.Number(visible=mode == PIXELIZE_FILTER)
        ]

    def on_video_model_change(self,
                              model_type: str,
                              vid_input: str):
        self.sam_inf.init_video_inference_state(vid_input=vid_input, model_type=model_type)
        frames = get_frames_from_dir(vid_dir=TEMP_DIR)
        initial_frame, max_frame_index = frames[0], (len(frames)-1)
        return [
            ImagePrompter(label="Prompt image with Box & Point", value=initial_frame),
            gr.Slider(label="Frame Index", value=0, interactive=True, step=1, minimum=0, maximum=max_frame_index)
        ]

    @staticmethod
    def on_frame_change(frame_idx: int):
        temp_dir = TEMP_DIR
        frames = get_frames_from_dir(vid_dir=temp_dir)
        selected_frame = frames[frame_idx]
        return ImagePrompter(label=f"Prompt image with Box & Point", value=selected_frame)

    @staticmethod
    def on_prompt_change(prompt: Dict):
        image, points = prompt["image"], prompt["points"]
        return gr.Image(label="Preview", value=image)

    def launch(self):
        _mask_hparams = self.default_hparams["mask_hparams"]

        with self.demo:
            md_header = gr.Markdown(HEADER, elem_id="md_header")

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
                    btn_open_folder.click(fn=lambda: open_folder(OUTPUT_PSD_DIR),
                                          inputs=None, outputs=None)
                    dd_input_modes.change(fn=self.on_mode_change,
                                          inputs=[dd_input_modes],
                                          outputs=[img_input, img_input_prompter, acc_mask_hparams])

                with gr.TabItem("Pixelize Filter"):
                    with gr.Column():
                        file_vid_input = gr.File(label="Input Video", file_types=IMAGE_FILE_EXT + VIDEO_FILE_EXT)
                        with gr.Row(equal_height=True):
                            with gr.Column(scale=9):
                                with gr.Row():
                                    vid_frame_prompter = ImagePrompter(label="Prompt image with Box & Point", type='pil',
                                                                       interactive=True, scale=5)
                                    img_preview = gr.Image(label="Preview", interactive=False, scale=5)

                                sld_frame_selector = gr.Slider(label="Frame Index", interactive=False)

                            with gr.Column(scale=1):
                                dd_models = gr.Dropdown(label="Model", value=DEFAULT_MODEL_TYPE,
                                                        choices=self.sam_inf.available_models)
                                dd_filter_mode = gr.Dropdown(label="Filter Modes", interactive=True,
                                                             value=self.default_filter,
                                                             choices=self.filter_modes)
                                cp_color_picker = gr.ColorPicker(label="Solid Color", interactive=True,
                                                                 visible=self.default_filter == COLOR_FILTER,
                                                                 value=self.default_color)
                                nb_pixel_size = gr.Number(label="Pixel Size", interactive=True, minimum=1,
                                                          visible=self.default_filter == PIXELIZE_FILTER,
                                                          value=self.default_pixel_size)
                                btn_generate_preview = gr.Button("GENERATE PREVIEW")

                    with gr.Row():
                        btn_generate = gr.Button("GENERATE", variant="primary")
                    with gr.Row():
                        vid_output = gr.Video(label="Output Video", interactive=False)
                        with gr.Column():
                            output_file = gr.File(label="Downloadable Output File", scale=9)
                            btn_open_folder = gr.Button("📁\nOpen Output folder", scale=1)

                    file_vid_input.change(fn=self.on_video_model_change,
                                          inputs=[dd_models, file_vid_input],
                                          outputs=[vid_frame_prompter, sld_frame_selector])
                    dd_models.change(fn=self.on_video_model_change,
                                     inputs=[dd_models, file_vid_input],
                                     outputs=[vid_frame_prompter, sld_frame_selector])
                    sld_frame_selector.change(fn=self.on_frame_change,
                                              inputs=[sld_frame_selector],
                                              outputs=[vid_frame_prompter],)
                    dd_filter_mode.change(fn=self.on_filter_mode_change,
                                          inputs=[dd_filter_mode],
                                          outputs=[cp_color_picker,
                                                   nb_pixel_size])

                    preview_params = [vid_frame_prompter, dd_filter_mode, sld_frame_selector, nb_pixel_size,
                                      cp_color_picker]
                    btn_generate_preview.click(fn=self.sam_inf.add_filter_to_preview,
                                               inputs=preview_params,
                                               outputs=[img_preview])
                    btn_generate.click(fn=self.sam_inf.create_filtered_video,
                                       inputs=preview_params,
                                       outputs=[vid_output, output_file])
                    btn_open_folder.click(fn=lambda: open_folder(OUTPUT_FILTER_DIR), inputs=None, outputs=None)

        self.demo.queue().launch(
            inbrowser=self.args.inbrowser,
            share=self.args.share
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=MODELS_DIR,
                        help='Model directory for segment-anything-2')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                        help='Output directory for the results')
    parser.add_argument('--inbrowser', type=bool, default=True, nargs='?', const=True,
                        help='Whether to automatically start Gradio app or not')
    parser.add_argument('--share', type=bool, default=False, nargs='?', const=True,
                        help='Whether to create a public link for the app or not')
    parser.add_argument('--theme', type=str, default=DEFAULT_THEME, help='Gradio Blocks theme')
    args = parser.parse_args()

    demo = App(args=args)
    demo.launch()
