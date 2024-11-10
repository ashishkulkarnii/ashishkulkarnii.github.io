---
title: "Image Editor"
excerpt: "Image editing command-line with implementations of greyscale, Gaussian blurring, masking based on colour range (HSV), etc. from scratch using OpenCV, NumPy, Pillow and Matplotlib."
collection: portfolio
permalink: "projects/image-editor"
---

This was a semester 4 Linear Algebra (UE20MA251) project.
The project contains a number of image manipulation techniques implemented from scratch.
These functionalities were combined to form:
- a command-line interface. Clone [this repository](https://github.com/ashishkulkarnii/image-editing-cli) to try it out.
- A [Streamlit app](https://image-editing.streamlit.app/).

## UI

The UI, hosted on [Streamlit](https://streamlit.io/), is a simple way to interact with the same functionalities implemented in the command-line app.
You can see how the UI looks in the following screenshots:

Original image | Brightened image
-- | --
![Original image](/images/image-editor-streamlit/1.png) | ![Brightness +73](/images/image-editor-streamlit/2.png)

## Command-line

### Functionalities

- greyscale conversion
- inverting
- solarizing
- contrast sdjusting
- resizing
- brightness adjusting
- gamma correction
- color pop
- mean blur
- gaussian blur
- color name in English to BGR

### Usage

#### General Commands:
* load &lt;filename>.&lt;extension>
* save [&lt;filename>] [&lt;extension>]
* exit [save]
* show [&lt;filename>.&lt;extension>]
* undo
* redo

`Note: [] means optional and all values are taken as float unless mentioned otherwise.`

#### Image Manipulation Commands:
* greyscale
* invert
* solarize &lt;"<" or ">"> &lt;threshold value from 0 to 255>
* contrast &lt;value from -100 to 100>
* resize &lt;new number (integer) of rows> &lt;new number (integer) of columns>
* brightness &lt;value from -100 to 100>
* gamma correction &lt;gamma value>
* color pop &lt;color name in English> [invert]
* mean blur &lt;kernel size (integer)>
* gaussian blur &lt;kernel size (integer)> [&lt;sigma value, default sigma = 1>]
* bgr &lt;color name in English>
