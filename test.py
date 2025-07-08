import ollama
from yyolo import detect
import json

path = 'a.jpeg'
image = None
try:
    if path:
        image = detect(path)
        if image:
            print("Image processed successfully.")
        else:
            print("No image detected or processed.")
except Exception as e:
    print(f"An error occurred: {e}")

def main(image: list):
    response = ollama.chat(
        model="gemma3:latest",
        messages=[
            {
                'role': "user",
                'content': (
                    "Identify the below output from an object detection model and generate a coherent, meaningful description of the image.\n"
                    f"{json.dumps(image, indent=2)}"
                ),
            }
        ]
    )
    return response['message']['content']

if __name__ == "__main__":
    if image:
        description = main(image)
        print(description)
    else:
        print("No image data to describe.")