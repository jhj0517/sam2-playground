# sam2-playground
Gradio based playground Web UI using [facebook/segment-anything-2](https://github.com/facebookresearch/segment-anything-2) models.

### Online Demos

[![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jhj0517/sam2-playground/blob/master/notebooks/sam2_playground.ipynb)
[![huggingface](https://img.shields.io/badge/Hugging%20Face-%F0%9F%A4%97%20Spaces-orange?logo=huggingface&labelColor=FFFFE0)](https://huggingface.co/spaces/jhj0517/sam2-playground)

## Feature
- Add filters to segment parts of a video with pixelated or solid color

<table>
  <tr>
    <td align="center"><strong>Pixelize the girl's face</strong></td>
  </tr>
  <tr>
    <td>
      <video controls autoplay loop src="https://github.com/user-attachments/assets/c5758970-dc15-4bc8-a918-8d3e8e44a73a" muted="false"></video>
    </td>
  </tr>
</table>


- Divide segmentation parts into layers and save them as PSD files.

<table>
  <tr>
    <td align="center"><strong>Divide clothes into layers and save as PSD file</strong></td>
  </tr>
  <tr>
    <td style="text-align: center;">
        <img src="https://github.com/jhj0517/sam2-playground/blob/master/docs/example_psd_file.png" alt="Example_PSD">
    </td>
  </tr>
</table>

See [PROMPT_GUIDE.md](https://github.com/jhj0517/sam2-playground/blob/master/docs/PROMPT_GUIDE.md) to see how to prompt the segmentation parts of the image.

# Installation and Running
### Prerequisites
To run this Web UI, you need these prerequisites. If you don't have them, please install them in the following links :

- `git` : https://git-scm.com/downloads
- `python=>3.10` : https://www.python.org/downloads/ 
- `FFmpeg` : https://ffmpeg.org/download.html
 
After installing FFmpeg, **make sure to add the `FFmpeg/bin` folder to your system PATH!** <br>
And for CUDA, if you're not using an Nvidia GPU and CUDA 12.4, edit the [requirements.txt](https://github.com/jhj0517/sam2-playground/blob/master/requirements.txt) to match your environment.

### Option 1 : Running with Shell / Batch Scripts
There's a set of shell / batch scripts for installation and running. 

1. Download `sam2-playground.zip` with the file corresponding to your OS from [sam2-playground-portable.zip](https://github.com/jhj0517/sam2-playground/releases/tag/v1.0.0) and extract its contents. 
2. Run `install.bat` or `install.sh` to install dependencies. (This will create a `venv` directory and install dependencies there.)
3. Start WebUI with `start-webui.bat` or `start-webui.sh` 
4. To update, run `update.bat` or `update.sh` 

### Option 2: Docker
1. Clone the repository
```
git clone https://github.com/jhj0517/sam2-playground.git
```
3. Build the image ( Image is about ~6 GB )
```
docker compose build
```
3. Run the container
```
docker compose up
```
4. Connect to `localhost:7860` with your browser.

If needed, update [`docker-compose.yaml`](https://github.com/jhj0517/sam2-playground/blob/master/docker-compose.yaml) to match your environments.

## Todo 🗓
- [ ] Support `change()` API for `gradio_image_prompter` and automatically generate preview for video predictor
