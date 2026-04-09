# main.py
import torch
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from tqdm import tqdm
import json 
import random
import config
import models
import reporting
from utils import build_grad_transfer_matrix
import pandas as pd  
from prompt_analyzer import analyze_prompt
import perturbation 
import gradient_analysis

def load_prompts_from_datasets(dataset_dir="datasets", dataset_files=None):
    prompts = []
    if dataset_files is None:
        dataset_files = ['advbench520.json']

    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory '{dataset_dir}' not found")
        return []

    for file_name in dataset_files:
        file_path = os.path.join(dataset_dir, file_name)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data:
                        if 'goal' in item and isinstance(item['goal'], str) and item['goal']:
                            prompts.append(item['goal'])
                print(f"Loaded {len(data)} prompts from {file_name}.")
            except Exception as e:
                print(f"Error reading or parsing {file_name}: {e}")
        else:
            print(f"Warning: Dataset file {file_path} not found")
            
    return prompts


def main():
    args = config.parse_args()

    # Optionally set visible CUDA devices from --device, e.g., cuda:0 -> 0.
    if args.device and args.device.startswith("cuda:"):
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device.split(":", 1)[1]

    if torch.cuda.is_available():
        torch.zeros(1, device="cuda")
        print("CUDA context initialized.")

    torch.manual_seed(42)
    random.seed(42)
    dataset_files = ['advbench520.json']
    # dataset_files = ['harmbench400.json']
    # dataset_files = ['wildjailbreak.json']


    dataset_tag = "_".join([d.replace('.json', '') for d in dataset_files])
    model_name = os.path.basename(args.model_path)
    method_tag = "M2" if args.guard_model_path else "M1"
    
    args.output_dir = os.path.join(args.output_dir, f"{model_name}_{dataset_tag}_{method_tag}")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"All results will be saved to: {args.output_dir}")

    
    filename_suffix = f"{dataset_tag}_{method_tag}"
    print(f"Filename identifier for this run: {filename_suffix}")

    llm_model, llm_tokenizer, guard_model, guard_tokenizer = models.load_models(args)
    
    if guard_model:
        print("\n--- Building gradient transfer matrix ---")
        transfer_matrix = build_grad_transfer_matrix(guard_tokenizer, llm_tokenizer, guard_model, llm_model).to(args.device)

    print("\n--- Loading Prompts from datasets ---")
    prompts_to_analyze = load_prompts_from_datasets(dataset_dir="datasets", dataset_files=dataset_files)
    
    
    # ========= Select number to load ============
    # prompts_to_analyze = prompts_to_analyze[:10]

    if not args.index_path:

        all_baseline_data = [] 
        llm_model.zero_grad()
        for prompt in tqdm(prompts_to_analyze, desc="Analyzing all Prompts"):

            if not guard_model:
                raise ValueError("guard_model is required. analyze_prompt_first has been removed.")

            llm_response_text, generation_kwargs = analyze_prompt(
                prompt, llm_model, llm_tokenizer, guard_model, guard_tokenizer, args, transfer_matrix
            )

            all_baseline_data.append({
                    "prompt": prompt,
                    "response": llm_response_text,
                    "gen_kwargs": generation_kwargs
                })
        S_grads ={}
        for name, param in llm_model.named_parameters():
            if param.grad is not None:
                S_grads[name] = param.grad.detach().clone().cpu()
            else: # Some parameters might not have gradients
                S_grads[name] = None
        save_dir = f"{args.output_dir}/saved_grad"
        os.makedirs(save_dir, exist_ok=True)
        torch.save(S_grads, f"{save_dir}/{model_name}-safe_grads_{method_tag}.pt")
        gradient_dfs = gradient_analysis.analyze_gradients(llm_model)
        gradient_dfs['full_analysis']['prompt'] = prompt

        print("\n\n--- Saving detailed analysis results for all Prompts ---")
        final_gradient_df = pd.concat(gradient_dfs, ignore_index=True)
        agg_grad_path = os.path.join(args.output_dir, "detailed_gradient_results_all_prompts.csv")
        final_gradient_df.to_csv(agg_grad_path, index=False, encoding='utf-8-sig')
        print(f"Detailed gradient analysis results for all prompts saved to: {agg_grad_path}")


        print("\n\n--- Starting calculation of average gradient information for all Prompts ---")
        # 1. Average - Absolute norm of all parameters (by parameter name)
        avg_params_abs = final_gradient_df.groupby('param_name')['grad_l2_norm'].mean().reset_index()
        avg_params_abs = avg_params_abs.sort_values(by='grad_l2_norm', ascending=False)
        # 2. Average - Relative norm of all parameters (by parameter name)
        avg_params_rel = final_gradient_df.groupby('param_name')['relative_grad_norm'].mean().reset_index()
        avg_params_rel = avg_params_rel.sort_values(by='relative_grad_norm', ascending=False)
        # 3. Average - Absolute norm aggregated by layer
        # First calculate sum for each layer within each prompt, then average these sums across prompts
        avg_layers_abs_sum = final_gradient_df[final_gradient_df['layer'] != 'N/A'] \
            .groupby(['prompt', 'layer'])['grad_l2_norm'].sum().reset_index() \
            .groupby('layer')['grad_l2_norm'].mean().reset_index()
        avg_layers_abs_sum = avg_layers_abs_sum.rename(columns={'grad_l2_norm': 'avg_aggregated_abs_norm'})
        # 4. Average - Relative norm aggregated by layer
        avg_layers_rel_sum = final_gradient_df[final_gradient_df['layer'] != 'N/A'] \
            .groupby(['prompt', 'layer'])['relative_grad_norm'].sum().reset_index() \
            .groupby('layer')['relative_grad_norm'].mean().reset_index()
        avg_layers_rel_sum = avg_layers_rel_sum.rename(columns={'relative_grad_norm': 'avg_aggregated_rel_norm'})
        # 5. Average - Absolute norm aggregated by component (same as above)
        avg_components_abs_sum = final_gradient_df.groupby(['prompt', 'component'])['grad_l2_norm'].sum().reset_index() \
            .groupby('component')['grad_l2_norm'].mean().reset_index()
        avg_components_abs_sum = avg_components_abs_sum.rename(columns={'grad_l2_norm': 'avg_aggregated_abs_norm'})
        # 6. Average - Relative norm aggregated by component (same as above)
        avg_components_rel_sum = final_gradient_df.groupby(['prompt', 'component'])['relative_grad_norm'].sum().reset_index() \
            .groupby('component')['relative_grad_norm'].mean().reset_index()
        avg_components_rel_sum = avg_components_rel_sum.rename(columns={'relative_grad_norm': 'avg_aggregated_rel_norm'})
        # 7. Print Top-10 results for each section
        print("\n\n" + "="*30 + " Average Gradient Analysis Top-10 Summary " + "="*30)
        print("\n--- [1] Top-10 Parameters (By Average Absolute Norm) ---")
        print(avg_params_abs.head(10).to_string())
        print("\n--- [2] Top-10 Parameters (By Average Relative Norm) ---")
        print(avg_params_rel.head(10).to_string())
        print("\n--- [3] Top-10 Layers (By Average Aggregated Absolute Norm) ---")
        print(avg_layers_abs_sum.sort_values('avg_aggregated_abs_norm', ascending=False).head(10).to_string())
        print("\n--- [4] Top-10 Layers (By Average Aggregated Relative Norm) ---")
        print(avg_layers_rel_sum.sort_values('avg_aggregated_rel_norm', ascending=False).head(10).to_string())
        print("\n--- [5] Top-10 Components (By Average Aggregated Absolute Norm) ---")
        print(avg_components_abs_sum.sort_values('avg_aggregated_abs_norm', ascending=False).head(10).to_string())
        print("\n--- [6] Top-10 Components (By Average Aggregated Relative Norm) ---")
        print(avg_components_rel_sum.sort_values('avg_aggregated_rel_norm', ascending=False).head(10).to_string())
        print("="*80 + "\n")
        # 8. Summarize and Plot
        avg_gradient_dfs = {
            "avg_params_abs": avg_params_abs,
            "avg_params_rel": avg_params_rel,
            "avg_layers_abs": avg_layers_abs_sum,
            "avg_layers_rel": avg_layers_rel_sum,
            "avg_components_abs": avg_components_abs_sum,
            "avg_components_rel": avg_components_rel_sum
        }
        reporting.generate_comprehensive_avg_gradient_plots(avg_gradient_dfs, args.output_dir, filename_suffix)


        # Top-K parameter verification experiment
    print("\n\n--- Starting perturbation verification experiment ---")
    test_prompt=prompts_to_analyze
    # test_prompt = random.sample(prompts_to_analyze, 100)
    with torch.no_grad(): 
        perturbation.perturb_and_evaluate(llm_model, llm_tokenizer, args, test_prompt)


if __name__ == "__main__":
    main()