# sam2-playground
Gradio based Playground Web UI using [facebook/segment-anything-2](https://github.com/facebookresearch/segment-anything-2) models.

## Online Demos
<div>
    <a href="https://colab.research.google.com/github/jhj0517/sam2-playground/blob/master/notebooks/sam2_playground.ipynb">
        <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" style="display:inline-block;">
    </a>
</div>

## Feature
- Add filters to segment parts of a video with pixelated or solid color

<table>
  <tr>
    <td>Pixelize the girl's face</td>
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
    <td>Divide clothes into layers and save as PSD file</td>
  </tr>
  <tr>
    <td>
        <img src="https://github.com/jhj0517/sam2-playground/blob/master/docs/example_psd_file.png" alt="Example_PSD">
    </td>
  </tr>
</table>

See PROMPT_GUIDE.md to see how to prompt the segmentation parts of the image.

# Installation and Running
### Prerequisites
To run this Web UI, you need these prerequisites. If you don't have them, please install them in the following links :

- `git` : https://git-scm.com/downloads
- `python => 3.10` : https://www.python.org/downloads/ 
- `FFmpeg` : https://ffmpeg.org/download.html
 
After installing FFmpeg, **make sure to add the `FFmpeg/bin` folder to your system PATH!** <br>
And for CUDA, if you're not using an Nvidia GPU and CUDA 12.4, edit the [requirements.txt](https://github.com/jhj0517/sam2-playground/blob/master/requirements.txt) to match your environment.

### Running with Shell / Batch Scripts
There's a set of shell / batch scripts for installation and running. 

1. Download `sam2-playground.zip` with the file corresponding to your OS from [sam2-playground-portable.zip]() and extract its contents. 
2. Run `install.bat` or `install.sh` to install dependencies. (This will create a `venv` directory and install dependencies there.)
3. Start WebUI with `start-webui.bat` or `start-webui.sh` 
4. To update the WebUI, run `update.bat` or `update.sh`

### Docker


## Todo
- [ ] Support `change()` API for `gradio_image_prompter` and automatically generate preview for video predictor
