# smolvlm_image_describer.py

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq # Changed from AutoModelForImageTextToText for Idefics2
from PIL import Image
import requests # For fetching image from URL

class SmolVLMImageDescriber:
    """
    A class to describe an image using a SmolVLM-like multimodal model.
    It takes an image and a text prompt, and returns a text description.
    This example uses a model like Idefics2, which follows a similar pattern.
    """

    def __init__(self, model_name="HuggingFaceM4/idefics2-8b-base", device=None):
        """
        Initialize the SmolVLM Image Describer.

        Args:
            model_name (str): The Hugging Face model identifier.
                              (e.g., "HuggingFaceM4/idefics2-8b-base" or a future SmolVLM equivalent)
            device (str, optional): Device to run inference on ('cpu', 'cuda', 'mps').
                                    Defaults to 'cuda' if available, else 'mps', else 'cpu'.
        """
        self.model_name = model_name

        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
        print(f"Using device: {self.device}")

        try:
            self.processor = AutoProcessor.from_pretrained(model_name)
            # For Idefics2, torch_dtype is important for memory.
            # Flash attention 2 can be used if available and dependencies are met.
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16 if self.device != "cpu" else torch.float32,
                # _attn_implementation="flash_attention_2" # Optional: if supported and installed
            ).to(self.device)
            self.model.eval()  # Set model to evaluation mode
            print(f"Model {model_name} loaded successfully on {self.device}.")
        except Exception as e:
            print(f"Error loading model or processor: {e}")
            raise

    def describe_image(self, image_input,
                       prompt_text="Describe this image in detail, focusing on the main subject and any notable actions.",
                       max_new_tokens=256):
        """
        Generates a text description for a given image based on a prompt.

        Args:
            image_input (str or PIL.Image.Image): Path to a local image, URL of an image, or a PIL Image object.
            prompt_text (str): The text prompt to guide the image description.
            max_new_tokens (int): The maximum number of new tokens to generate for the description.

        Returns:
            str: The generated text description of the image, or an error message.
        """
        pil_image = None
        try:
            if isinstance(image_input, str):
                if image_input.startswith("http://") or image_input.startswith("https://"):
                    pil_image = Image.open(requests.get(image_input, stream=True).raw).convert("RGB")
                else:
                    pil_image = Image.open(image_input).convert("RGB")
            elif isinstance(image_input, Image.Image):
                pil_image = image_input.convert("RGB")
            else:
                return "Error: Invalid image_input type. Must be a file path, URL, or PIL.Image object."
        except Exception as e:
            return f"Error loading image: {e}"

        if pil_image is None:
            return "Error: Could not process the image input."

        # Construct the messages payload for the chat template.
        # For Idefics2 and similar models, you provide an image placeholder in the content.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},  # Placeholder for the image
                    {"type": "text", "text": prompt_text},
                ]
            }
        ]

        try:
            # Apply the chat template to get the formatted prompt string (without tokenizing yet)
            # This will insert special tokens around the image placeholder and text.
            text_prompt_with_placeholder = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True, # Ensures the prompt is ready for generation
                tokenize=False # We get the string, processor call below will tokenize
            )

            # Process the text prompt and the PIL image together
            # The processor will handle tokenizing text and preprocessing the image.
            inputs = self.processor(
                text=text_prompt_with_placeholder,
                images=[pil_image], # Pass the image as a list
                return_tensors="pt"
            ).to(self.device)
            
            # For bfloat16 models, ensure inputs match if not already handled by processor
            if self.model.dtype == torch.bfloat16:
                inputs = {k: v.to(torch.bfloat16) if torch.is_floating_point(v) else v for k, v in inputs.items()}


            # Generate text
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            
            # Decode the generated tokens
            # We need to slice off the input tokens from the generated_ids for batch_decode
            # if the model is an encoder-decoder or if generate includes input_ids.
            # For many vision-language models, generated_ids are just the new tokens.
            # If not, generated_ids = generated_ids[:, inputs['input_ids'].shape[1]:] might be needed.
            # However, AutoModelForVision2Seq.generate and processor.batch_decode usually handle this.

            generated_texts = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )
            
            return generated_texts[0].strip()

        except Exception as e:
            return f"Error during model inference or processing: {e}"

# Example Usage (similar to ViTDeepFakeDetector's main block)
if __name__ == "__main__":
    print("Initializing SmolVLMImageDescriber...")
    # You might need to log in to Hugging Face CLI if the model requires it: `huggingface-cli login`
    try:
        # Using a smaller, faster Idefics2 variant for quicker testing if needed:
        # describer = SmolVLMImageDescriber(model_name="HuggingFaceM4/idefics2-8b-base")
        # Or use a more capable one if you have the resources.
        # For this example, let's try to keep it manageable.
        # If "HuggingFaceM4/idefics2-8b-base" is too large, consider smaller alternatives
        # or ensure you have enough RAM/VRAM.
        describer = SmolVLMImageDescriber(model_name="HuggingFaceM4/idefics2-8b-base") # Requires ~16GB VRAM for bfloat16
                                          # For CPU, it will be very slow and require >32GB RAM.
    except Exception as e:
        print(f"Failed to initialize describer: {e}")
        print("Please ensure you have sufficient resources and are logged into Hugging Face if required.")
        exit()

    # Test with a common image URL
    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/bee.jpg"
    print(f"\nDescribing image from URL: {image_url}")
    description1 = describer.describe_image(image_url, prompt_text="What is this insect and what is it doing?")
    print("Description 1:")
    print(description1)

    # Test with another prompt
    print(f"\nDescribing image from URL with a different prompt: {image_url}")
    description2 = describer.describe_image(image_url, prompt_text="Describe the colors and patterns on this bee.")
    print("Description 2:")
    print(description2)

    # Example of how to use a local file (create a dummy file or use your own)
    local_image_path = "test_bee_image.jpg" # Make sure this image exists or change the path
    try:
        # Download the image for local testing if it doesn't exist
        import os
        if not os.path.exists(local_image_path):
            print(f"Downloading image to {local_image_path} for local test...")
            img_data = requests.get(image_url).content
            with open(local_image_path, 'wb') as handler:
                handler.write(img_data)
        
        if os.path.exists(local_image_path):
            print(f"\nDescribing local image: {local_image_path}")
            description3 = describer.describe_image(local_image_path, "What is in the background of this image?")
            print("Description 3:")
            print(description3)
        else:
            print(f"\nSkipping local image test: {local_image_path} not found.")

    except Exception as e:
        print(f"Error in local image test: {e}")