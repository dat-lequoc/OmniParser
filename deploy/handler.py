import runpod
import torch
from PIL import Image
import io
import base64
import os
from utils import get_som_labeled_img, check_ocr_box, get_caption_model_processor, get_yolo_model

class OmniParserPredictor:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Initializing models on {self.device}")
        
        # Initialize models
        self.model_path = 'weights/icon_detect_v1_5/model_v1_5.pt'
        self.som_model = get_yolo_model(self.model_path)
        self.som_model.to(self.device)
        
        # Initialize caption model
        self.caption_model_processor = get_caption_model_processor(
            model_name="florence2",
            model_name_or_path="weights/icon_caption_florence",
            device=self.device
        )
        print("Models initialized successfully")

    def predict(self, event):
        """
        Handler function for processing images through the model.
        Expects base64 encoded image in the input.
        """
        try:
            # Get the input image from the event
            image_data = base64.b64decode(event["input"]["image"])
            
            # Create temporary directory if it doesn't exist
            os.makedirs("temp", exist_ok=True)
            temp_image_path = "temp/temp_image.png"
            
            # Save temporary image
            with open(temp_image_path, "wb") as f:
                f.write(image_data)
            
            # Process image
            image = Image.open(temp_image_path)
            image_rgb = image.convert('RGB')
            
            # Configure box overlay
            box_overlay_ratio = max(image.size) / 3200
            draw_bbox_config = {
                'text_scale': 0.8 * box_overlay_ratio,
                'text_thickness': max(int(2 * box_overlay_ratio), 1),
                'text_padding': max(int(3 * box_overlay_ratio), 1),
                'thickness': max(int(3 * box_overlay_ratio), 1),
            }
            
            BOX_TRESHOLD = 0.05
            
            # Run OCR
            ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
                temp_image_path,
                display_img=False,
                output_bb_format='xyxy',
                goal_filtering=None,
                easyocr_args={'paragraph': False, 'text_threshold': 0.8},
                use_paddleocr=True
            )
            text, ocr_bbox = ocr_bbox_rslt
            
            # Get labeled image and parsed content
            _, _, parsed_content_list = get_som_labeled_img(
                temp_image_path,
                self.som_model,
                BOX_TRESHOLD=BOX_TRESHOLD,
                output_coord_in_ratio=True,
                ocr_bbox=ocr_bbox,
                draw_bbox_config=draw_bbox_config,
                caption_model_processor=self.caption_model_processor,
                ocr_text=text,
                use_local_semantics=True,
                iou_threshold=0.7,
                scale_img=False,
                batch_size=128
            )
            
            # Clean up temporary file
            os.remove(temp_image_path)
            
            return {"status": "success", "data": parsed_content_list}
            
        except Exception as e:
            return {"status": "error", "error": str(e)}

# Initialize the predictor
predictor = OmniParserPredictor()

def handler(event):
    """
    RunPod handler function that uses the predictor
    """
    return predictor.predict(event)

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})