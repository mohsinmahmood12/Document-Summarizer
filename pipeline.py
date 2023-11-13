from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
from preprocess import file_preprocessing
import torch
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load model and tokenizer
checkpoint = "MBZUAI/LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map='auto', torch_dtype=torch.float32)

# Initialize pipeline
summarization_pipe = pipeline('summarization', model=base_model, tokenizer=tokenizer)

def llm_pipeline(filepath, max_length=500, min_length=50):
    """
    Summarizes the content of a given file.

    Parameters:
    filepath (str): Path to the file to be summarized.
    max_length (int): Maximum length of the summary.
    min_length (int): Minimum length of the summary.

    Returns:
    str: The summarized text.
    """
    # Validate input file
    if not os.path.exists(filepath):
        logging.error(f"File not found: {filepath}")
        return None

    try:
        input_text = file_preprocessing(filepath)
        result = summarization_pipe(input_text, max_length=max_length, min_length=min_length)
        return result[0]['summary_text']
    except Exception as e:
        logging.error(f"Error during summarization: {e}")
        return None