import requests
from PIL import Image
import io

def test_parser(image_path: str):
    # Prepare the image file
    with open(image_path, 'rb') as f:
        files = {'file': f}
        
    # Make the request
    response = requests.post(
        'http://localhost:8000/parse_image',
        files=files
    )
    
    return response.json()['parsed_content']

# Test with an image
result = test_parser('screenshot.png')
print(result)