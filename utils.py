# utils.py
# (Assuming you already have the get_template_prompt_embeddings function)
import torch

def get_template_embeddings(embed_weights, device=None):
    t1 = [
        220,
        128000,
        128006,
        882,
        128007,
        271,
        6396,
        25,
        4343,
        422,
        1070,
        374,
        20451,
        2262,
        304,
        364,
        1502,
        6,
        6743,
        304,
        21633,
        4184,
        1057,
        7296,
        4947,
        449,
        279,
        3770,
        11306,
        382,
        33619,
        16841,
        47083,
        30952,
        36868,
        356,
        67696,
        397,
        50,
        16,
        25,
        30555,
        306,
        78008,
        627,
        50,
        17,
        25,
        11842,
        20198,
        25947,
        306,
        78008,
        627,
        50,
        18,
        25,
        6834,
        78008,
        627,
        50,
        19,
        25,
        9576,
        7943,
        385,
        7709,
        627,
        50,
        20,
        25,
        3979,
        27279,
        627,
        50,
        21,
        25,
        9984,
        1534,
        55820,
        627,
        50,
        22,
        25,
        19406,
        627,
        50,
        23,
        25,
        77956,
        8825,
        627,
        50,
        24,
        25,
        2314,
        42710,
        3357,
        47664,
        627,
        50,
        605,
        25,
        66912,
        627,
        50,
        806,
        25,
        10323,
        11529,
        2227,
        627,
        50,
        717,
        25,
        39767,
        9059,
        627,
        50,
        1032,
        25,
        58601,
        627,
        50,
        975,
        25,
        6247,
        83593,
        46784,
        627,
        27,
        4794,
        47083,
        30952,
        36868,
        356,
        67696,
        1363,
        33619,
        16841,
        3501,
        73326,
        3579,
        1363,
        1502,
        25,
    ]
    t2 = [
        271,
        27,
        4794,
        3501,
        73326,
        3579,
        1363,
        61524,
        701,
        7296,
        15813,
        369,
        27785,
        3247,
        48395,
        2724,
        1984,
        304,
        279,
        3485,
        10652,
        512,
        482,
        5629,
        1584,
        2011,
        1373,
        364,
        19193,
        6,
        477,
        364,
        39257,
        24482,
        482,
        1442,
        20451,
        11,
        264,
        2132,
        1584,
        2011,
        2997,
        264,
        32783,
        73792,
        1160,
        315,
        34521,
        11306,
        13,
        128009,
        128006,
        78191,
        128007,
        271,
    ]
    if device is None:
        device = embed_weights.device
    t1, t2 = torch.tensor(t1).to(device), torch.tensor(t2).to(device)
    one_hot_1 = torch.zeros(
        t1.shape[0], embed_weights.shape[0], device=device, dtype=embed_weights.dtype
    )
    one_hot_1.scatter_(
        1,
        t1.unsqueeze(1),
        torch.ones(one_hot_1.shape[0], 1, device=device, dtype=embed_weights.dtype),
    )
    template_embeds_1 = (one_hot_1 @ embed_weights).unsqueeze(0)

    one_hot_2 = torch.zeros(
        t2.shape[0], embed_weights.shape[0], device=device, dtype=embed_weights.dtype
    )
    one_hot_2.scatter_(
        1,
        t2.unsqueeze(1),
        torch.ones(one_hot_2.shape[0], 1, device=device, dtype=embed_weights.dtype),
    )
    template_embeds_2 = (one_hot_2 @ embed_weights).unsqueeze(0)
    return template_embeds_1.detach(), template_embeds_2.detach()

def get_template_prompt_embeddings(
    embedding_weights, 
    input_embeds, 
    device,
):
    template_embeds_1, template_embeds_2 = get_template_embeddings(embedding_weights, device = device)

    full_embeds = torch.cat([template_embeds_1, input_embeds, template_embeds_2], dim=1)

    full_embeds = full_embeds.to(device)
    # print("full_embeds.shape: ", full_embeds.shape)
    return full_embeds

def extract_content(tag, text):
    start_idx = text.find(tag)
    if start_idx == -1: return None
    content_after_tag = text[start_idx + len(tag):].strip()
    parts = content_after_tag.split()
    if tag == "#thescore:":
        try:
            score_str = parts[0].strip().replace('.', '')
            assert score_str.isdigit()
            return int(score_str)
        except (AssertionError, IndexError, ValueError):
             print(f"  - Warning: Could not parse integer score from '{parts}', returning 0.")
             return 0
    else:
        end_idx = text.find("#", start_idx + 1)
        return content_after_tag if end_idx == -1 else content_after_tag[:end_idx].strip()

def build_grad_transfer_matrix(guard_tok, llm_tok, guard_model, llm_model, dtype=torch.float32):  # Build gradient transfer matrix to align tokens between guard model and llm model
    """Build Gradient Transfer Matrix"""  # Function description: Map gradients from the guard model to the LLM's embedding space
    G = guard_model.get_input_embeddings().weight.size(0)  # Guard model embedding size (vocab size, rows)
    L = llm_model.get_input_embeddings().weight.size(0)    # LLM model embedding size (vocab size, cols)
    rows, cols, vals = [], [], []                          # Initialize lists for sparse matrix indices and values
    for gid in range(G):                                   # Iterate through each token id in guard model
        if gid >= guard_tok.vocab_size: continue           # Skip if id exceeds tokenizer vocab size
        tok = guard_tok.convert_ids_to_tokens(gid)         # Convert token id back to string
        if tok is None: continue                           # Skip if conversion fails (e.g., special tokens)
        if tok[0] in ("Ġ", "_"):
            tok = " " + tok[1:]          
        lid_seq = llm_tok(tok, add_special_tokens=False).input_ids  # Re-tokenize using LLM's tokenizer (may map to multiple sub-words)
        if not lid_seq: continue                           # Skip if tokenization yields empty result
        w = 1.0 / len(lid_seq)                             # Calculate average weight (split evenly if one guard token maps to multiple llm tokens)
        for lid in lid_seq:                                # Iterate through all mapped llm tokens
            rows.append(gid)                               # Record matrix row index (guard token id)
            cols.append(lid)                               # Record matrix column index (llm token id)
            vals.append(w)                                 # Record weight value for this mapping
    return torch.sparse_coo_tensor(                        # Construct sparse COO tensor (Sparse Matrix)
        indices=torch.tensor([rows, cols]),                # 2D tensor of row/col indices
        values=torch.tensor(vals, dtype=dtype),            # Values corresponding to non-zero elements (mapping weights)
        size=(G, L),                                       # Sparse matrix size: (guard_vocab_size, llm_vocab_size)
    ).coalesce()                                           # Coalesce duplicate indices (sum values if mapped to same position multiple times)