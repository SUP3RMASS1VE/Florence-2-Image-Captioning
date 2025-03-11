# Florence-2 Model Demo README

This project demonstrates the capabilities of Microsoft's [Florence-2 Model](https://huggingface.co/microsoft/Florence-2-large) through a simple web interface built using Gradio. Florence-2 is a large vision-language model capable of various image and text generation tasks, such as object detection, captioning, and grounding. This demo allows users to interact with these capabilities by uploading images and selecting from various tasks.

## Key Features

- **Image Captioning**: Automatically generate captions for the uploaded images.
- **Detailed and More Detailed Captioning**: Get captions with additional details based on image context.
- **Object Detection**: Detect and visualize objects in images with bounding boxes.
- **Region-based Grounding**: Generate captions based on specific regions in the image.
- **Text-to-Image Interaction**: Perform phrase grounding or OCR on image regions.

## Installation

To run this demo locally, you need to have Python 3.7+ installed, along with the necessary dependencies.

### Step 1: Clone the Repository

```bash
git clone [https://github.com/SUP3RMASS1VE/Florence-2-Image-Captioning]
cd Florence-2-Image-Captioning
```

### Step 2: Install the Required Libraries

Create a virtual environment (optional but recommended):

```bash
python -m venv env
env\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

The required dependencies include:

- `gradio`: For the web interface.
- `transformers`: For the model and processor.
- `torch`: To handle model execution on CUDA.
- `PIL`, `matplotlib`, and `numpy`: For image manipulation.

### Step 3: Run the Demo

To run the demo locally, use the following command:

```bash
python app.py
```

This will start a local Gradio web server, accessible in your browser.

## Features and Usage

### Tasks Available

1. **Caption**: Generate a short caption for the image.
2. **Detailed Caption**: Get a more detailed description.
3. **More Detailed Caption**: Generate an even more elaborate description.
4. **Caption + Grounding**: Caption the image and identify grounded regions.
5. **Object Detection**: Detect and label objects within the image.
6. **Dense Region Caption**: Generate captions for regions of interest within the image.
7. **Region Proposal**: Generate proposed regions in the image for analysis.
8. **Referring Expression Segmentation**: Segment parts of the image based on textual input.
9. **Open Vocabulary Detection**: Detect objects based on open vocabulary.
10. **Region to Category**: Classify regions in the image into categories.
11. **OCR**: Extract textual information from the image using optical character recognition (OCR).
12. **OCR with Region**: Perform OCR with region localization for better accuracy.

### Gradio Interface

The app is powered by [Gradio](https://gradio.app/) and provides an intuitive interface:

- **Upload Image**: Upload the image you want to analyze.
- **Task Selection**: Choose one of the tasks (e.g., Caption, Object Detection).
- **Submit**: After selecting the task, click the "Run" button to process the image and get the output.

### Example

1. Upload an image of an object or scene.
2. Choose a task such as "Caption" or "Object Detection."
3. The model processes the image and returns a relevant output (e.g., a caption or bounding boxes for detected objects).

## Model Loading and Usage

### Available Models

- `microsoft/Florence-2-large-ft`
- `microsoft/Florence-2-large`
- `microsoft/Florence-2-base-ft`
- `microsoft/Florence-2-base`

These models are loaded and made available for different tasks based on the task's requirement. You can choose the model you want to use based on your needs for performance and precision.

### Running the Model

The core logic of processing involves:
1. Loading the image using `PIL`.
2. Preprocessing the image and text inputs.
3. Feeding them into the Florence-2 model using `transformers`.
4. Returning the generated outputs, which could be text (captions, detections) or visual annotations (bounding boxes, polygons).

## Customization

### Change Task Prompts
Modify the `task_prompt` for different model behaviors such as captioning or detecting objects. The tasks are defined with specific keywords, which are then mapped to Florence-2's capabilities.

### Update Model/Processor
To update or switch to a different Florence model, modify the `models` and `processors` dictionary in the code to point to a new model checkpoint.

## Troubleshooting

- **CUDA Issues**: Ensure you have a compatible CUDA setup if running the models on a GPU.
- **Memory**: Florence-2 models can be large, so you may need a system with sufficient RAM and GPU memory for inference.
- **Model Downloading**: If the model fails to download, ensure that you have internet access and the required Hugging Face credentials for downloading.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Credits

- The models used in this demo are provided by Microsoft and are hosted on [Hugging Face](https://huggingface.co/).
- This demo was built using the Gradio library, a great tool for rapid prototyping of machine learning models.

## Contributing

Feel free to fork this repository and submit pull requests if you'd like to contribute enhancements, bug fixes, or features!

---

For more information, please visit the [Florence-2 demo page](https://huggingface.co/microsoft/Florence-2-large).

