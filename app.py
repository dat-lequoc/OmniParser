from typing import Optional
import gradio as gr
import numpy as np
import torch
from PIL import Image
import io
import base64
import os
import pandas as pd
from utils import check_ocr_box, get_yolo_model, get_caption_model_processor, get_som_labeled_img

MARKDOWN = """
# OmniParser for Pure Vision Based General GUI Agent 🔥
<div>
    <a href="https://arxiv.org/pdf/2408.00203">
        <img src="https://img.shields.io/badge/arXiv-2408.00203-b31b1b.svg" alt="Arxiv" style="display:inline-block;">
    </a>
</div>

OmniParser is a screen parsing tool to convert general GUI screen to structured elements. 
"""

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def process(
    image_input,
    box_threshold,
    iou_threshold,
    use_paddleocr,
    imgsz,
    icon_process_batch_size,
) -> Optional[tuple]:
    image_save_path = 'imgs/saved_image_demo.png'
    os.makedirs('imgs', exist_ok=True)
    
    image_input.save(image_save_path)
    image = Image.open(image_save_path)
    box_overlay_ratio = image.size[0] / 3200
    draw_bbox_config = {
        'text_scale': 0.8 * box_overlay_ratio,
        'text_thickness': max(int(2 * box_overlay_ratio), 1),
        'text_padding': max(int(3 * box_overlay_ratio), 1),
        'thickness': max(int(3 * box_overlay_ratio), 1),
    }

    ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
        image_save_path, 
        display_img=False, 
        output_bb_format='xyxy', 
        goal_filtering=None, 
        easyocr_args={'paragraph': False, 'text_threshold':0.9}, 
        use_paddleocr=use_paddleocr
    )
    text, ocr_bbox = ocr_bbox_rslt

    dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
        image_save_path,
        yolo_model,
        BOX_TRESHOLD=box_threshold,
        output_coord_in_ratio=True,
        ocr_bbox=ocr_bbox,
        draw_bbox_config=draw_bbox_config,
        caption_model_processor=caption_model_processor,
        ocr_text=text,
        iou_threshold=iou_threshold,
        imgsz=imgsz,
        batch_size=icon_process_batch_size
    )

    # Create visualization image
    image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
    
    # Create DataFrame and CSV
    df = pd.DataFrame(parsed_content_list)
    df['ID'] = range(len(df))
    csv_path = "parsed_elements.csv"
    df.to_csv(csv_path, index=False)
    
    # Create text summary
    text_summary = '\n'.join([f'type: {x["type"]}, content: {x["content"]}, interactivity: {x["interactivity"]}' for x in parsed_content_list])
    
    return image, text_summary, csv_path

# Load models
icon_detect_model = 'weights/icon_detect/best.pt'
icon_caption_model = 'florence2'

yolo_model = get_yolo_model(model_path=icon_detect_model)
yolo_model.to(DEVICE)

if icon_caption_model == 'florence2':
    caption_model_processor = get_caption_model_processor(
        model_name="florence2", 
        model_name_or_path="weights/icon_caption_florence",
        device=DEVICE
    )
elif icon_caption_model == 'blip2':
    caption_model_processor = get_caption_model_processor(
        model_name="blip2", 
        model_name_or_path="weights/icon_caption_blip2",
        device=DEVICE
    )

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown(MARKDOWN)
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type='pil', label='Upload image')
            box_threshold = gr.Slider(
                label='Box Threshold', 
                minimum=0.01, 
                maximum=1.0, 
                step=0.01, 
                value=0.05
            )
            iou_threshold = gr.Slider(
                label='IOU Threshold', 
                minimum=0.01, 
                maximum=1.0, 
                step=0.01, 
                value=0.1
            )
            use_paddleocr = gr.Checkbox(label='Use PaddleOCR', value=False)
            imgsz = gr.Slider(
                label='Icon Detect Image Size', 
                minimum=640, 
                maximum=3200, 
                step=32, 
                value=1920
            )
            icon_process_batch_size = gr.Slider(
                label='Icon Process Batch Size', 
                minimum=1, 
                maximum=256, 
                step=1, 
                value=64
            )
            submit_button = gr.Button(value='Submit', variant='primary')
        
        with gr.Column():
            image_output = gr.Image(type='pil', label='Image Output')
            text_output = gr.Textbox(
                label='Parsed screen elements', 
                placeholder='Text Output'
            )
            file_output = gr.File(label="Download Parsed Elements CSV")

    submit_button.click(
        fn=process,
        inputs=[
            image_input,
            box_threshold,
            iou_threshold,
            use_paddleocr,
            imgsz,
            icon_process_batch_size
        ],
        outputs=[image_output, text_output, file_output]
    )

if __name__ == "__main__":
    demo.launch(share=True, server_port=7861, server_name='0.0.0.0')
