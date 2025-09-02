"""
FastVLM-7B Remote Zoo Model for FiftyOne
Implements Apple's FastVLM-7B vision-language model as a FiftyOne zoo model
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Union

import numpy as np
import torch
from PIL import Image
from huggingface_hub import snapshot_download

import fiftyone as fo
from fiftyone import Model, SamplesMixin
from fiftyone.core.labels import Classification, Classifications
from fiftyone.operators import types

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

logger = logging.getLogger(__name__)

# Constants
MODEL_ID = "apple/FastVLM-7B"
IMAGE_TOKEN_INDEX = -200  # Special token for image placeholder

DEFAULT_VQA_SYSTEM_PROMPT = """You are a helpful assistant. You provide clear and concise answers to questions about images. Report answers in natural language text in English."""

# Operation modes and their default prompts
OPERATIONS = {
    "vqa": DEFAULT_VQA_SYSTEM_PROMPT,
}


def get_device():
    """Get the appropriate device for model inference."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

class FastVLM(SamplesMixin, Model):
    """
    A FiftyOne model for running Apple's FastVLM-7B vision-language model.
    
    This model performs Visual Question Answering (VQA) on images.
    """
    
    def __init__(
        self,
        model_path: str,
        prompt: str = None,
        system_prompt: str = None,
        temperature: float = 0.7,
        max_new_tokens: int = 512,
        top_p: float = 0.90,
        top_k: int = 50,
        **kwargs
    ):
        """
        Initialize the FastVLM model.
        
        Args:
            model_path: Path to the downloaded model
            prompt: Default prompt/question for all images
            system_prompt: Custom system prompt (optional)
            temperature: Generation temperature (0.1-2.0)
            max_new_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
        """
        self._fields = {}
        
        self.model_path = model_path
        self.prompt = prompt or "What is in this image?"
        self._custom_system_prompt = system_prompt
        
        # Generation parameters
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.top_k = top_k
        
        # Get device
        self.device = get_device()
        logger.info(f"Using device: {self.device}")
        
        # Determine torch dtype based on device
        if self.device == "cuda" and torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            # Enable bfloat16 on Ampere+ GPUs (compute capability 8.0+)
            if capability[0] >= 8:
                self.torch_dtype = torch.bfloat16
            else:
                self.torch_dtype = torch.float16
        else:
            self.torch_dtype = torch.float32
        
        # Model loading kwargs
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": self.torch_dtype,
            "device_map": "auto" if self.device == "cuda" else None,
        }
        
        # Load tokenizer
        logger.info("Loading tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # Load model
        logger.info("Loading model")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs
        )
        
        if self.device != "cuda":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        logger.info("Model loaded successfully")
    
    @property
    def needs_fields(self):
        """A dict mapping model-specific keys to sample field names."""
        return self._fields
    
    @needs_fields.setter
    def needs_fields(self, fields):
        self._fields = fields
    
    def _get_field(self):
        """Get the prompt field from needs_fields."""
        if "prompt_field" in self.needs_fields:
            prompt_field = self.needs_fields["prompt_field"]
        else:
            prompt_field = next(iter(self.needs_fields.values()), None)
        
        return prompt_field
    
    @property
    def media_type(self):
        """Returns the media type for the model."""
        return "image"
    
    @property
    def system_prompt(self):
        """Return custom system prompt if set, otherwise return default."""
        return self._custom_system_prompt if self._custom_system_prompt is not None else DEFAULT_VQA_SYSTEM_PROMPT
    
    @system_prompt.setter
    def system_prompt(self, value):
        """Set a custom system prompt."""
        self._custom_system_prompt = value
    
    def _predict(self, image: Image.Image, sample=None):
        """Process a single image through the model and return VQA response."""
        # Get prompt from sample field if available
        current_prompt = self.prompt
        if sample is not None and self._get_field() is not None:
            field_value = sample.get_field(self._get_field())
            if field_value is not None:
                current_prompt = str(field_value)
        
        # Build the user content with system prompt and question
        user_content = f"{self.system_prompt}\n\nQuestion: {current_prompt}"
        
        # Build chat messages with <image> placeholder
        messages = [
            {"role": "user", "content": f"<image>\n{user_content}"}
        ]
        
        # Render to string (not tokens) so we can place <image> exactly
        rendered = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        
        # Split at image token
        pre, post = rendered.split("<image>", 1)
        
        # Tokenize the text around the image token
        pre_ids = self.tokenizer(pre, return_tensors="pt", add_special_tokens=False).input_ids
        post_ids = self.tokenizer(post, return_tensors="pt", add_special_tokens=False).input_ids
        
        # Splice in the IMAGE token id at the placeholder position
        img_tok = torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)
        input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1).to(self.model.device)
        attention_mask = torch.ones_like(input_ids, device=self.model.device)
        
        # Process image through model's vision tower
        px = self.model.get_vision_tower().image_processor(images=image, return_tensors="pt")["pixel_values"]
        px = px.to(self.model.device, dtype=self.torch_dtype)
        
        # Generate response
        with torch.inference_mode():
            out = self.model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                images=px,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode response
        response = self.tokenizer.decode(out[0], skip_special_tokens=True)
        
        # Extract just the assistant's response if present
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()
        
        # Return as Classification with the response as the label
        return response.strip()
    
    def predict(self, image, sample=None):
        """
        Process an image with the model for Visual Question Answering.
        
        Args:
            image: PIL Image, numpy array, or path to an image
            sample: Optional FiftyOne sample containing the image
            
        Returns:
            fo.Classification containing the VQA response as the label
        """
        # Convert input to PIL Image
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        else:
            image = image.convert("RGB")
        
        return self._predict(image, sample)