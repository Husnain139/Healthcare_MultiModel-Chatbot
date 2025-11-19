#Step1: Setup GROQ API key
import os
import base64
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def encode_image(image_path):
    """Encode image file to base64 string"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        raise FileNotFoundError(f"Image file '{image_path}' not found!")


def analyze_image_with_query(query, encoded_image, model):
    """Analyze an image using GROQ's multimodal LLM"""
    
    # Check if API key exists
    if not os.environ.get("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY environment variable not set!")
    
    # Create GROQ client
    client = Groq()
    
    # Prepare messages
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": query
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}",
                    },
                },
            ],
        }
    ]
    
    # Get response
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model
    )
    
    return chat_completion.choices[0].message.content


if __name__ == "__main__":
    # Configuration
    image_path = "acne.jpg"
    query = "Is there something wrong with my face?"
    model = "meta-llama/llama-4-scout-17b-16e-instruct"
    # Alternative models:
    # model = "meta-llama/llama-4-maverick-17b-128e-instruct"
    # model = "llama-3.2-90b-vision-preview" #Deprecated
    
    try:
        encoded_image = encode_image(image_path)
        result = analyze_image_with_query(query, encoded_image, model)
        print("\n=== AI Doctor Analysis ===")
        print(result)
        print("\n" + "="*30)
    except Exception as e:
        print(f"Error: {e}")
