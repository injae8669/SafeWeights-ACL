# prompt_analyzer.py
import torch
import torch.nn.functional as F
# Import analysis module
import gradient_analysis

def analyze_prompt(prompt_text, llm_model, llm_tokenizer, guard_model, guard_tokenizer, args, transfer_matrix):
    print(f"\n\n{'='*20} Analyzing Prompt: '{prompt_text[:50]}...' {'='*20}")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."}, 
        {"role": "user", "content": prompt_text},
    ]
    input_ids = llm_tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(llm_model.device)
    # input_ids = llm_tokenizer(
    #         messages,
    #         return_tensors="pt"
    #     ).to(llm_model.device)
    # terminators = [llm_tokenizer.eos_token_id, llm_tokenizer.convert_tokens_to_ids("<|eot_id|>")] # eot_id is specific to Llama models
    terminators = [llm_tokenizer.eos_token_id]

    # --- Prepare kwargs for perturbation experiment ---
    generation_kwargs = {
        'input_ids': input_ids, 
        'max_new_tokens': args.max_new_tokens,
        'eos_token_id': terminators, 
        'do_sample': False,
        'pad_token_id': llm_tokenizer.eos_token_id
    }
    # messages = [
    #     {"role": "system", "content": "You are a helpful assistant."}, 
    #     {"role": "user", "content": prompt_text},
    # ]

    # # 1. Convert chat structure to plain text prompt
    # prompt_text_formatted = (
    #     f"System: {messages[0]['content']}\n"
    #     f"User: {messages[1]['content']}\n"
    #     "Assistant:"
    # )

    # # 2. tokenizer only accepts strings
    # inputs = llm_tokenizer(
    #     prompt_text_formatted,
    #     return_tensors="pt"
    # ).to(llm_model.device)

    # input_ids = inputs["input_ids"]

    # # -------- 3. terminators same as original --------
    # terminators = [llm_tokenizer.eos_token_id] # eot_id is specific to Llama models

    # # -------- 4. generation_kwargs --------
    # generation_kwargs = {
    #     'input_ids': input_ids, 
    #     'max_new_tokens': args.max_new_tokens,
    #     'eos_token_id': terminators, 
    #     'do_sample': False,
    #     'pad_token_id': llm_tokenizer.eos_token_id,
    #     'repetition_penalty': 1.1,     # prevent repetition
    #     'no_repeat_ngram_size': 4,  
    # }
    # --- (End) ---

    with torch.no_grad():
        outputs = llm_model.generate(**generation_kwargs)    
    
    model_outputs = llm_model(outputs)
    logits = model_outputs.logits
    prompt_len = input_ids.shape[-1]

    if outputs.shape[1] <= prompt_len:
        print(f"Warning: Model generated no new content. Skipping this prompt.")
        return None, None

    response_logits = logits[:, prompt_len - 1:-2, :]
    response_ids = outputs[:, prompt_len:-1]
    llm_response_text = llm_tokenizer.decode(response_ids[0], skip_special_tokens=True).strip()
    print(f"Test LLM Generated Response: '{llm_response_text}'")

    if not llm_response_text.strip():
        print("Analysis aborted because generated response is empty.")
        return None, None

    # 4. Calculate gradients and backpropagate
    # 4.1 First calculate guard gradient (guard model gradient w.r.t llm_response)
    guard_grad = gradient_analysis.grad_guard_response(guard_model, guard_tokenizer, llm_response_text, args.device)
    # Multiply guard gradient by projection matrix
    llm_grad = guard_grad.to(torch.float32) @ transfer_matrix
    
    seq_len = min(response_logits.shape[1], llm_grad.shape[1])
    if seq_len == 0:
        print("Analysis aborted because gradient and logits sequence lengths do not match.")
        return None, None
        
    response_logits_sliced = response_logits[:, :seq_len, :]
    llm_grad_sliced = llm_grad[:, :seq_len, :]
    
    soft_response_probs = F.gumbel_softmax(response_logits_sliced, tau=0.3, hard=False)

    # ====== Verification: Check simulated one-hot probs ========
    max_probs, max_ids = soft_response_probs.max(dim=-1)
    print("//" * 50)
    print("testmodel gumbel-softmax prob is ", max_probs[0])
    print("//" * 50)
    # ========================================

    loss = - (llm_grad_sliced.detach() * soft_response_probs).sum()
    
    # llm_model.zero_grad()
    loss.backward()
    print("Backpropagation complete.")

    return llm_response_text, generation_kwargs