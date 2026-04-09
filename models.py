# models.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_models(args):
    """Load LLM and Guard models and tokenizers."""
    print("--- Loading Models ---")
    dtype = torch.bfloat16 if "cuda" in args.device else torch.float32

    # Load Guard Model
    guard_model = None
    guard_tokenizer = None
    if args.guard_model_path:
        print("Loading Guard model...")
        guard_model = AutoModelForCausalLM.from_pretrained(args.guard_model_path, torch_dtype=dtype, device_map="auto")
        guard_model.eval()
        guard_tokenizer = AutoTokenizer.from_pretrained(args.guard_model_path)
        print("Guard model loaded.")

    # Load test-LLM Model
    print("\nLoading test-LLM model...")
    llm_model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype, device_map="auto")
    llm_tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    print("test-LLM model loaded.")
    
    return llm_model, llm_tokenizer, guard_model, guard_tokenizer