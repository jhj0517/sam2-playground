## Image Prompter
There’s an image prompter UI that you can use in the app.

![image-prompter](https://github.com/jhj0517/sam2-playground/blob/master/docs/image_prompter_screenshot.png)

### Prompt with points or boxes.

To prompt the segmentation area as desired, **left-click** on the part you want to use as the point. You'll see a blue dot in the image prompter. <br>
Let's use the point (Blue dot) to segment the child.

![point](https://github.com/jhj0517/sam2-playground/blob/master/docs/prompt_with_point.png)

Or alternatively, You can draw box around the child.

![box](https://github.com/jhj0517/sam2-playground/blob/master/docs/prompt_with_box.png)

### Prompt with point & box combination
What if we want to prompt only on the face of the child instead of the entire child? <br>
Then you can draw a box around the child's face and **emphasize that area** with the point within the box.

![comb](https://github.com/jhj0517/sam2-playground/blob/master/docs/prompt_with_box_and_point_combination.png)

Remember that you can't use the box and point combination prompt in SAM-2. You can only use a single combination at a time.

 ### Prompt with negative points
You can also use **negative points** (red dots) to segment a more specific part. <br>
Use the **wheel click** to prompt negative points.  <br>
To segment only the face, place positive points (blue dots) on the face and a negative point on the body. <br>

![ng](https://github.com/jhj0517/sam2-playground/blob/master/docs/prompt_with_negative_points.png)


### Known bugs in the Web UI
In the "Pixelize Filter" tab, you should reset the prompt by pressing the eraser button on the image prompter whenever you change the frame index, as the previous prompt remains even after you've changed the frame index. <br>

![eraser](https://github.com/jhj0517/sam2-playground/blob/master/docs/eraser_button.png)

This may occurs because the [gradio-image-prompter](https://github.com/PhyscalX/gradio-image-prompter) does not support the `change()` API yet—it's on the Todo list now.


