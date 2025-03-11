import gradio as gr
import warnings

# Filter out specific deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")
warnings.filterwarnings("ignore", category=FutureWarning, module="timm.models.layers")
# Filter out the warning about Florence2LanguageForConditionalGeneration generative capabilities
warnings.filterwarnings("ignore", message=".*Florence2LanguageForConditionalGeneration has generative capabilities.*")

from transformers import AutoProcessor, AutoModelForCausalLM

import requests
import copy

from PIL import Image, ImageDraw, ImageFont 
import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import random
import numpy as np


models = {
    'microsoft/Florence-2-large-ft': AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-large-ft', trust_remote_code=True).to("cuda").eval(),
    'microsoft/Florence-2-large': AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-large', trust_remote_code=True).to("cuda").eval(),
    'microsoft/Florence-2-base-ft': AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-base-ft', trust_remote_code=True).to("cuda").eval(),
    'microsoft/Florence-2-base': AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-base', trust_remote_code=True).to("cuda").eval(),
}

processors = {
    'microsoft/Florence-2-large-ft': AutoProcessor.from_pretrained('microsoft/Florence-2-large-ft', trust_remote_code=True),
    'microsoft/Florence-2-large': AutoProcessor.from_pretrained('microsoft/Florence-2-large', trust_remote_code=True),
    'microsoft/Florence-2-base-ft': AutoProcessor.from_pretrained('microsoft/Florence-2-base-ft', trust_remote_code=True),
    'microsoft/Florence-2-base': AutoProcessor.from_pretrained('microsoft/Florence-2-base', trust_remote_code=True),
}


DESCRIPTION = "# [Florence-2 Demo](https://huggingface.co/microsoft/Florence-2-large)"

colormap = ['blue','orange','green','purple','brown','pink','gray','olive','cyan','red',
            'lime','indigo','violet','aqua','magenta','coral','gold','tan','skyblue']

def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return Image.open(buf)

def run_example(task_prompt, image, text_input=None, model_id='microsoft/Florence-2-large'):
    model = models[model_id]
    processor = processors[model_id]
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )
    return parsed_answer

def plot_bbox(image, data):
    fig, ax = plt.subplots()
    ax.imshow(image)
    for bbox, label in zip(data['bboxes'], data['labels']):
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))
    ax.axis('off')
    return fig

def draw_polygons(image, prediction, fill_mask=False):

    draw = ImageDraw.Draw(image)
    scale = 1
    for polygons, label in zip(prediction['polygons'], prediction['labels']):
        color = random.choice(colormap)
        fill_color = random.choice(colormap) if fill_mask else None
        for _polygon in polygons:
            _polygon = np.array(_polygon).reshape(-1, 2)
            if len(_polygon) < 3:
                print('Invalid polygon:', _polygon)
                continue
            _polygon = (_polygon * scale).reshape(-1).tolist()
            if fill_mask:
                draw.polygon(_polygon, outline=color, fill=fill_color)
            else:
                draw.polygon(_polygon, outline=color)
            draw.text((_polygon[0] + 8, _polygon[1] + 2), label, fill=color)
    return image

def convert_to_od_format(data):
    bboxes = data.get('bboxes', [])
    labels = data.get('bboxes_labels', [])
    od_results = {
        'bboxes': bboxes,
        'labels': labels
    }
    return od_results

def draw_ocr_bboxes(image, prediction):
    scale = 1
    draw = ImageDraw.Draw(image)
    bboxes, labels = prediction['quad_boxes'], prediction['labels']
    for box, label in zip(bboxes, labels):
        color = random.choice(colormap)
        new_box = (np.array(box) * scale).tolist()
        draw.polygon(new_box, width=3, outline=color)
        draw.text((new_box[0]+8, new_box[1]+2),
                  "{}".format(label),
                  align="right",
                  fill=color)
    return image

def process_image(image, task_prompt, text_input=None, model_id='microsoft/Florence-2-large'):
    image = Image.fromarray(image)  # Convert NumPy array to PIL Image
    if task_prompt == 'Caption':
        task_prompt = '<CAPTION>'
        results = run_example(task_prompt, image, model_id=model_id)
        return results, None
    elif task_prompt == 'Detailed Caption':
        task_prompt = '<DETAILED_CAPTION>'
        results = run_example(task_prompt, image, model_id=model_id)
        return results, None
    elif task_prompt == 'More Detailed Caption':
        task_prompt = '<MORE_DETAILED_CAPTION>'
        results = run_example(task_prompt, image, model_id=model_id)
        return results, None
    elif task_prompt == 'Caption + Grounding':
        task_prompt = '<CAPTION>'
        results = run_example(task_prompt, image, model_id=model_id)
        text_input = results[task_prompt]
        task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
        results = run_example(task_prompt, image, text_input, model_id)
        results['<CAPTION>'] = text_input
        fig = plot_bbox(image, results['<CAPTION_TO_PHRASE_GROUNDING>'])
        return results, fig_to_pil(fig)
    elif task_prompt == 'Detailed Caption + Grounding':
        task_prompt = '<DETAILED_CAPTION>'
        results = run_example(task_prompt, image, model_id=model_id)
        text_input = results[task_prompt]
        task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
        results = run_example(task_prompt, image, text_input, model_id)
        results['<DETAILED_CAPTION>'] = text_input
        fig = plot_bbox(image, results['<CAPTION_TO_PHRASE_GROUNDING>'])
        return results, fig_to_pil(fig)
    elif task_prompt == 'More Detailed Caption + Grounding':
        task_prompt = '<MORE_DETAILED_CAPTION>'
        results = run_example(task_prompt, image, model_id=model_id)
        text_input = results[task_prompt]
        task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
        results = run_example(task_prompt, image, text_input, model_id)
        results['<MORE_DETAILED_CAPTION>'] = text_input
        fig = plot_bbox(image, results['<CAPTION_TO_PHRASE_GROUNDING>'])
        return results, fig_to_pil(fig)
    elif task_prompt == 'Object Detection':
        task_prompt = '<OD>'
        results = run_example(task_prompt, image, model_id=model_id)
        fig = plot_bbox(image, results['<OD>'])
        return results, fig_to_pil(fig)
    elif task_prompt == 'Dense Region Caption':
        task_prompt = '<DENSE_REGION_CAPTION>'
        results = run_example(task_prompt, image, model_id=model_id)
        fig = plot_bbox(image, results['<DENSE_REGION_CAPTION>'])
        return results, fig_to_pil(fig)
    elif task_prompt == 'Region Proposal':
        task_prompt = '<REGION_PROPOSAL>'
        results = run_example(task_prompt, image, model_id=model_id)
        fig = plot_bbox(image, results['<REGION_PROPOSAL>'])
        return results, fig_to_pil(fig)
    elif task_prompt == 'Caption to Phrase Grounding':
        task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
        results = run_example(task_prompt, image, text_input, model_id)
        fig = plot_bbox(image, results['<CAPTION_TO_PHRASE_GROUNDING>'])
        return results, fig_to_pil(fig)
    elif task_prompt == 'Referring Expression Segmentation':
        task_prompt = '<REFERRING_EXPRESSION_SEGMENTATION>'
        results = run_example(task_prompt, image, text_input, model_id)
        output_image = copy.deepcopy(image)
        output_image = draw_polygons(output_image, results['<REFERRING_EXPRESSION_SEGMENTATION>'], fill_mask=True)
        return results, output_image
    elif task_prompt == 'Region to Segmentation':
        task_prompt = '<REGION_TO_SEGMENTATION>'
        results = run_example(task_prompt, image, text_input, model_id)
        output_image = copy.deepcopy(image)
        output_image = draw_polygons(output_image, results['<REGION_TO_SEGMENTATION>'], fill_mask=True)
        return results, output_image
    elif task_prompt == 'Open Vocabulary Detection':
        task_prompt = '<OPEN_VOCABULARY_DETECTION>'
        results = run_example(task_prompt, image, text_input, model_id)
        bbox_results = convert_to_od_format(results['<OPEN_VOCABULARY_DETECTION>'])
        fig = plot_bbox(image, bbox_results)
        return results, fig_to_pil(fig)
    elif task_prompt == 'Region to Category':
        task_prompt = '<REGION_TO_CATEGORY>'
        results = run_example(task_prompt, image, text_input, model_id)
        return results, None
    elif task_prompt == 'Region to Description':
        task_prompt = '<REGION_TO_DESCRIPTION>'
        results = run_example(task_prompt, image, text_input, model_id)
        return results, None
    elif task_prompt == 'OCR':
        task_prompt = '<OCR>'
        results = run_example(task_prompt, image, model_id=model_id)
        return results, None
    elif task_prompt == 'OCR with Region':
        task_prompt = '<OCR_WITH_REGION>'
        results = run_example(task_prompt, image, model_id=model_id)
        output_image = copy.deepcopy(image)
        output_image = draw_ocr_bboxes(output_image, results['<OCR_WITH_REGION>'])
        return results, output_image
    else:
        return "", None  # Return empty string and None for unknown task prompts

css = """
  @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;900&family=Exo+2:wght@300;400;600;700&display=swap');
  
  #output {
    height: 500px; 
    overflow: auto; 
    border: 2px solid rgba(138, 43, 226, 0.7); 
    border-radius: 12px;
    box-shadow: 0 0 25px rgba(138, 43, 226, 0.5), inset 0 0 15px rgba(138, 43, 226, 0.3);
    background-color: rgba(13, 5, 30, 0.8);
    color: #ffffff;
    backdrop-filter: blur(5px);
  }
  
  .gradio-container {
    background: radial-gradient(ellipse at center, #0d0d2b 0%, #090920 50%, #050510 100%);
    background-image: url('https://images.unsplash.com/photo-1534796636912-3b95b3ab5986?q=80&w=2071&auto=format&fit=crop'), radial-gradient(ellipse at center, #0d0d2b 0%, #090920 50%, #050510 100%);
    background-blend-mode: overlay;
    background-size: cover;
    font-family: 'Exo 2', sans-serif;
    color: #ffffff;
    position: relative;
  }
  
  .gradio-container::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI0MDAiIGhlaWdodD0iNDAwIj48ZmlsdGVyIGlkPSJub2lzZSIgeD0iMCIgeT0iMCIgd2lkdGg9IjEwMCUiIGhlaWdodD0iMTAwJSI+PGZlVHVyYnVsZW5jZSB0eXBlPSJmcmFjdGFsTm9pc2UiIGJhc2VGcmVxdWVuY3k9IjAuNjUiIG51bU9jdGF2ZXM9IjMiIHN0aXRjaFRpbGVzPSJzdGl0Y2giIHJlc3VsdD0ibm9pc2UiLz48ZmVDb2xvck1hdHJpeCB0eXBlPSJtYXRyaXgiIHZhbHVlcz0iMCAwIDAgMCAwIDAgMCAwIDAgMCAwIDAgMCAwIDAgMCAwIDAgMC4xIDAiLz48L2ZpbHRlcj48cmVjdCB3aWR0aD0iNDAwIiBoZWlnaHQ9IjQwMCIgZmlsdGVyPSJ1cmwoI25vaXNlKSIgb3BhY2l0eT0iMC40Ii8+PC9zdmc+');
    opacity: 0.3;
    z-index: -1;
    pointer-events: none;
  }
  
  .gr-button {
    background: linear-gradient(90deg, #8a2be2 0%, #4b0082 100%) !important;
    border: none !important;
    color: white !important;
    border-radius: 30px !important;
    padding: 10px 20px !important;
    font-family: 'Orbitron', sans-serif !important;
    font-weight: bold !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    transition: all 0.3s ease;
    box-shadow: 0 0 15px rgba(138, 43, 226, 0.6) !important;
    position: relative;
    overflow: hidden;
  }
  
  .gr-button::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: linear-gradient(45deg, transparent, rgba(255, 255, 255, 0.1), transparent);
    transform: rotate(45deg);
    animation: shine 3s infinite;
  }
  
  @keyframes shine {
    0% { transform: translateX(-100%) rotate(45deg); }
    100% { transform: translateX(100%) rotate(45deg); }
  }
  
  .gr-button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 0 30px rgba(138, 43, 226, 0.8) !important;
  }
  
  .gr-form {
    border-radius: 16px;
    box-shadow: 0 0 30px rgba(0, 0, 0, 0.5);
    padding: 25px;
    background: rgba(13, 5, 30, 0.7);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(138, 43, 226, 0.3);
    position: relative;
    overflow: hidden;
  }
  
  .gr-form::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: radial-gradient(circle at 30% 40%, rgba(138, 43, 226, 0.1) 0%, transparent 60%);
    pointer-events: none;
  }
  
  h1, h2, h3 {
    font-family: 'Orbitron', sans-serif;
    background: linear-gradient(90deg, #e066ff, #8a2be2, #4b0082);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    text-shadow: 0 0 10px rgba(224, 102, 255, 0.5);
    letter-spacing: 1px;
    animation: glow 3s infinite alternate;
  }
  
  @keyframes glow {
    0% { text-shadow: 0 0 10px rgba(224, 102, 255, 0.5); }
    100% { text-shadow: 0 0 20px rgba(224, 102, 255, 0.8), 0 0 30px rgba(138, 43, 226, 0.6); }
  }
  
  .gr-input, .gr-dropdown {
    background-color: rgba(13, 5, 30, 0.7) !important;
    border: 2px solid rgba(138, 43, 226, 0.5) !important;
    border-radius: 10px !important;
    color: white !important;
    padding: 12px !important;
    font-family: 'Exo 2', sans-serif !important;
    box-shadow: 0 0 10px rgba(138, 43, 226, 0.2) !important;
    transition: all 0.3s ease;
  }
  
  .gr-input:focus, .gr-dropdown:focus {
    border-color: #e066ff !important;
    box-shadow: 0 0 15px rgba(224, 102, 255, 0.5) !important;
  }
  
  .gr-panel {
    border-radius: 16px;
    background: rgba(13, 5, 30, 0.7);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(138, 43, 226, 0.3);
    box-shadow: 0 0 20px rgba(0, 0, 0, 0.5);
  }
  
  .gr-box {
    border-radius: 12px;
    background: rgba(13, 5, 30, 0.6);
    border: 1px solid rgba(138, 43, 226, 0.3);
    box-shadow: inset 0 0 10px rgba(138, 43, 226, 0.2);
  }
  
  label {
    color: #e066ff !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    font-family: 'Exo 2', sans-serif !important;
    text-shadow: 0 0 5px rgba(224, 102, 255, 0.5);
  }
  
  .gr-radio {
    accent-color: #8a2be2 !important;
  }
  
  .gr-tab {
    background-color: rgba(13, 5, 30, 0.7) !important;
    color: white !important;
    border-radius: 8px 8px 0 0 !important;
    font-family: 'Orbitron', sans-serif !important;
    letter-spacing: 0.5px;
  }
  
  .gr-tab-selected {
    background: linear-gradient(90deg, #8a2be2 0%, #4b0082 100%) !important;
    color: white !important;
    font-weight: bold !important;
    box-shadow: 0 0 15px rgba(138, 43, 226, 0.5);
  }
  
  /* Add star-like dots to the background */
  .gradio-container::after {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    background-image: 
      radial-gradient(1px 1px at 10px 10px, rgba(255, 255, 255, 0.9), rgba(255, 255, 255, 0)),
      radial-gradient(1px 1px at 20px 50px, rgba(255, 255, 255, 0.9), rgba(255, 255, 255, 0)),
      radial-gradient(2px 2px at 30px 100px, rgba(255, 255, 255, 0.9), rgba(255, 255, 255, 0)),
      radial-gradient(1px 1px at 70px 130px, rgba(255, 255, 255, 0.9), rgba(255, 255, 255, 0)),
      radial-gradient(1px 1px at 90px 40px, rgba(255, 255, 255, 0.9), rgba(255, 255, 255, 0)),
      radial-gradient(1px 1px at 130px 80px, rgba(255, 255, 255, 0.9), rgba(255, 255, 255, 0)),
      radial-gradient(1px 1px at 160px 120px, rgba(255, 255, 255, 0.9), rgba(255, 255, 255, 0));
    background-repeat: repeat;
    background-size: 200px 200px;
  }
"""


single_task_list =[
    'Caption', 'Detailed Caption', 'More Detailed Caption', 'Object Detection',
    'Dense Region Caption', 'Region Proposal', 'Caption to Phrase Grounding',
    'Referring Expression Segmentation', 'Region to Segmentation',
    'Open Vocabulary Detection', 'Region to Category', 'Region to Description',
    'OCR', 'OCR with Region'
]

cascased_task_list =[
    'Caption + Grounding', 'Detailed Caption + Grounding', 'More Detailed Caption + Grounding'
]


def update_task_dropdown(choice):
    if choice == 'Cascased task':
        return gr.Dropdown(choices=cascased_task_list, value='Caption + Grounding')
    else:
        return gr.Dropdown(choices=single_task_list, value='Caption')



with gr.Blocks(css=css) as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Tab(label="Florence-2 Image Captioning"):
        with gr.Row():
            with gr.Column():
                input_img = gr.Image(label="Input Picture")
                model_selector = gr.Dropdown(choices=list(models.keys()), label="Model", value='microsoft/Florence-2-large')
                task_type = gr.Radio(choices=['Single task', 'Cascased task'], label='Task type selector', value='Single task')
                task_prompt = gr.Dropdown(choices=single_task_list, label="Task Prompt", value="Caption")
                task_type.change(fn=update_task_dropdown, inputs=task_type, outputs=task_prompt)
                text_input = gr.Textbox(label="Text Input (optional)")
                submit_btn = gr.Button(value="Submit")
            with gr.Column():
                output_text = gr.Textbox(label="Output Text")
                output_img = gr.Image(label="Output Image")

        submit_btn.click(process_image, [input_img, task_prompt, text_input, model_selector], [output_text, output_img])

demo.launch(server_name="127.0.0.1", server_port=7860, share=False, debug=True)
