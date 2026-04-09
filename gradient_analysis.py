import torch
import pandas as pd
from collections import defaultdict
from types import SimpleNamespace
from utils import get_template_prompt_embeddings 


def grad_guard_response(guard_model, guard_tokenizer, response, device):
    """Calculate gradient for Guard Model (Simplified, removed dependency on args)"""
    args = SimpleNamespace(guard_model="Llama-Guard") # Internal creation

    # ========= guard-model part (start) =========
    emb_weight = guard_model.get_input_embeddings().weight.to(device)
    init_ids = guard_tokenizer(response, return_tensors="pt").input_ids.to(device)[0]
    
    logits_mat = torch.full((1, init_ids.size(0), emb_weight.size(0)), -17.0, device=device, dtype=emb_weight.dtype)
    logits_mat[0, torch.arange(init_ids.size(0)), init_ids] = 0.0
    logits_mat = torch.nn.Parameter(logits_mat, requires_grad=True)

    # Simulate one-hot probabilities
    probs = torch.softmax(logits_mat, dim=-1)
    max_probs, max_ids = probs.max(dim=-1)
    print("//"*50)
    print("guardmodel softmax prob is ", max_probs[0])
    print("//"*50)
    
    embeds = torch.softmax(logits_mat, dim=-1) @ emb_weight
    
    full_embeds = get_template_prompt_embeddings(emb_weight, embeds, device)
    logits = guard_model(inputs_embeds=full_embeds).logits
    last_token_logits = logits[0, -1, :]
    
    loss_main = -last_token_logits[19193] # Increase logits for safe token, i.e., its corresponding probability
    # ========= guard-model part (end) =========
    
    loss_main.backward(retain_graph=True)
    guard_grad = logits_mat.grad.clone().detach()
    
    print(f"loss main is: {loss_main.item():.4f}")
    safe_score = torch.softmax(last_token_logits, dim=-1)[19193]
    print(f"safe score is: {safe_score.item():.4f}")
    
    return guard_grad

def get_component_name(param_name: str) -> str:
    """Extract fine-grained component name from parameter name (compatible with LLaMA / Qwen series)"""
    name = param_name.lower()

    # === Attention Related ===
    if "q_proj" in name: return "self_attn.q_proj"
    if "k_proj" in name: return "self_attn.k_proj"
    if "v_proj" in name: return "self_attn.v_proj"
    if "o_proj" in name: return "self_attn.o_proj"

    # === MLP Related ===
    if "gate_proj" in name: return "mlp.gate_proj"
    if "up_proj" in name: return "mlp.up_proj"
    if "down_proj" in name: return "mlp.down_proj"

    # === Norm Layers ===
    if "input_layernorm" in name or "ln_1" in name: return "input_layernorm"
    if "post_attention_layernorm" in name or "ln_2" in name: return "post_attention_layernorm"
    if "final_norm" in name or "ln_f" in name or "norm.weight" in name: return "final_norm"

    # === Embedding / Head ===
    if "embed_tokens" in name: return "embed_tokens"
    if "lm_head" in name: return "lm_head"

    return "other"

def analyze_gradients(model):
    print("\n--- Step 7: Deep Analysis of Model Gradients ---")
    analysis_data = []
    layer_component_grad_norms = defaultdict(lambda: defaultdict(float))

    for name, param in model.named_parameters():
        if param.grad is not None:

            grad_mean_abs = torch.mean(torch.abs(param.grad)).item()

            param_std = torch.std(param.data).item()
            
            relative_grad_norm = grad_mean_abs * param_std
            
            # Extract layer name and component name
            layer_name = "N/A"
            parts = name.split('.')
            if "layers" in parts:
                idx = parts.index("layers")
                if idx + 1 < len(parts) and parts[idx + 1].isdigit():
                    layer_name = f"layers.{parts[idx + 1]}"
            lower_name = name.lower()
            if (
                "embed" in lower_name or
                "norm" in lower_name or
                "bias" in lower_name or
                "mlp.gate.weight" in lower_name
            ):
                
                continue
            component_name = get_component_name(name)
    

            analysis_data.append({
                'param_name': name, 
                'grad_l2_norm': grad_mean_abs,
                'param_l2_norm': param_std, 
                'relative_grad_norm': relative_grad_norm,
                'layer': layer_name,
                'component': component_name
            })
            
            # Aggregate gradient norms per layer and per component
            if layer_name != "N/A":
                layer_component_grad_norms[layer_name][component_name] += grad_mean_abs

    df_analysis = pd.DataFrame(analysis_data)

    print("Gradient analysis complete. ✅")
    
    return {
        "full_analysis": df_analysis,        # Raw full analysis data
    }

