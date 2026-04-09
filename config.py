# config.py
import argparse

def parse_args():
    """Centrally manage and parse command-line arguments."""
    parser = argparse.ArgumentParser(description="LLMs Safety Analysis through Gradient and Perturbation")

    # --- Model Paths ---
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model.")
    parser.add_argument("--guard_model_path", type=str,
                        help="Path to the Llama Guard model.")

    # --- API Credentials ---
    parser.add_argument("--api_secret_key", type=str,
                        help="API key for the evaluation service.")
    parser.add_argument("--base_url", type=str,
                        help="Base URL for the evaluation service.")
                        
    # --- Generation and Analysis Parameters ---
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum number of new tokens to generate.")
    parser.add_argument("--noise_ratio", type=float, default=0.38,
                        help="Noise magnitude ratio for perturbation analysis.")
    parser.add_argument("--device", type=str,
                        help="Primary CUDA device to use.")

    parser.add_argument("--top_k_ratio", type=float, default=0.01,
                        help="The ratio for top-k selection.")
    
    parser.add_argument("--selection_mode", type=str, default="topk",
                        help="Selection mode: 'topk', 'random', or 'bottomk'.")

    parser.add_argument("--index_path", type=str, default=None,
                        help="Directory to load previously saved gradient information. If provided, gradient analysis is skipped.")

    parser.add_argument("--resort_k", type=float, default=None,
                        help="Select a subset from the loaded gradient information. "
                             "E.g., if original gradient info is top 1% (0.01), setting resort_k=0.1 selects the top 10% within that 0.01.")
    
    # --- Output Settings ---
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save analysis results (plots, csv files). ")

    args = parser.parse_args()
    return args