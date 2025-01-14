import requests
import base64
import json
import os
from typing import Dict, Any, Optional

class OmniParserClient:
    def __init__(self, api_key: str, endpoint_url: str = "https://api.runpod.ai/v2/42883k4i91accs/run"):
        """Initialize the OmniParser client
        
        Args:
            api_key (str): RunPod API key for authentication
            endpoint_url (str): The URL of the RunPod endpoint
        """
        self.endpoint_url = endpoint_url
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }

    def parse_image(self, image_path: str) -> Dict[str, Any]:
        """Send an image to the OmniParser service for parsing
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            Dict containing the parsing results
        """
        # Read and encode image
        with open(image_path, 'rb') as image_file:
            base64_image = base64.b64encode(image_file.read()).decode()
        
        # Prepare payload
        payload = {
            "input": {
                "image": base64_image
            }
        }
        
        # Send POST request
        try:
            response = requests.post(
                self.endpoint_url,
                json=payload,
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"status": "error", "error": str(e)}

def test_parser(image_path: str) -> None:
    """Test the OmniParser service with a single image
    
    Args:
        image_path (str): Path to the image file to test
    """
    api_key = os.getenv('RUNPOD_API_KEY')
    if not api_key:
        print("Error: RUNPOD_API_KEY environment variable not set")
        exit(1)
        
    client = OmniParserClient(api_key)
    
    print(f"Testing image: {image_path}")
    result = client.parse_image(image_path)
    
    if result["status"] == "success":
        print("✅ Parsing successful!")
        print("\nParsed content:")
        print(json.dumps(result["data"], indent=2))
    else:
        print("❌ Parsing failed!")
        print(f"Error: {result['error']}")

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Test the OmniParser API')
    parser.add_argument('image_path', help='Path to the image file to parse')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found: {args.image_path}")
        exit(1)
        
    test_parser(args.image_path)