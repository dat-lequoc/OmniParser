from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import torch
from utils import get_som_labeled_img, check_ocr_box, get_caption_model_processor, get_yolo_model

app = FastAPI()

# Global variables for models
device = 'cuda'
model_path = 'weights/icon_detect_v1_5/model_v1_5.pt'
BOX_TRESHOLD = 0.05

# Initialize models
som_model = get_yolo_model(model_path)
som_model.to(device)

caption_model_processor = get_caption_model_processor(
    model_name="florence2", 
    model_name_or_path="weights/icon_caption_florence", 
    device=device
)

@app.post("/parse_image")
async def parse_image(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Calculate box overlay ratio
    box_overlay_ratio = max(image.size) / 3200
    draw_bbox_config = {
        'text_scale': 0.8 * box_overlay_ratio,
        'text_thickness': max(int(2 * box_overlay_ratio), 1),
        'text_padding': max(int(3 * box_overlay_ratio), 1),
        'thickness': max(int(3 * box_overlay_ratio), 1),
    }

    # Process OCR
    ocr_bbox_rslt, _ = check_ocr_box(
        image, 
        display_img=False, 
        output_bb_format='xyxy', 
        goal_filtering=None,
        easyocr_args={'paragraph': False, 'text_threshold':0.8}, 
        use_paddleocr=True
    )
    text, ocr_bbox = ocr_bbox_rslt

    # Get labeled image and parsed content
    _, _, parsed_content_list = get_som_labeled_img(
        image,
        som_model,
        BOX_TRESHOLD=BOX_TRESHOLD,
        output_coord_in_ratio=True,
        ocr_bbox=ocr_bbox,
        draw_bbox_config=draw_bbox_config,
        caption_model_processor=caption_model_processor,
        ocr_text=text,
        use_local_semantics=True,
        iou_threshold=0.7,
        scale_img=False,
        batch_size=128
    )

    return {"parsed_content": parsed_content_list}