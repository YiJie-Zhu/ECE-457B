import torch
import numpy as np
import math

torch.manual_seed(123123)

# 4.1.a
def generate_token_embeddings(s):
    id_map = {}
    s_map = {}
    s_list = s.split(" ")
    
    # sort by alphabetical order
    s_list_sorted = list(dict.fromkeys(sorted(s.split(" "))))
    for i, s in enumerate(s_list_sorted):
        id_map[i] = s
        s_map[s] = i + 1

    T = len(s_list)
    tensor = []

    for s in s_list:
        tensor.append(s_map[s])

    tensor = torch.as_tensor(tensor)

    # create embedding with torch
    embedding = torch.nn.Embedding(T + 1, 16)

    output_embedding = embedding(tensor)
    print(f"Embedding of word: {output_embedding}")
    print(f"Shape of embedding: {output_embedding.shape}")

    return output_embedding

#4.1.b
def generate_contex(x, d_q, d_k, d_v):
    embeddings_np = x.detach().numpy()
    T = embeddings_np.shape[1]

    # generate q k and v
    q = torch.rand(d_q, T)
    k = torch.rand(d_k, T)
    v = torch.rand(d_v, T)

    q_x = torch.matmul(x, q.transpose(0, 1))
    k_x = torch.matmul(x, k.transpose(0, 1))
    v_x = torch.matmul(x, v.transpose(0, 1))

    w_x = torch.matmul(q_x, k_x.transpose(0, 1))

    alpha_x = torch.nn.functional.softmax(torch.div(w_x, math.sqrt(d_k)), dim=-1)

    context = torch.matmul(alpha_x, v_x)

    print(f"Context: {context}")
    print(f"Shape of single-head context: {context.shape}")

    return context

# 4.2
def generate_multi_contex(embeddings, d_q, d_k, d_v, h):

    c_multi = []

    for _ in range(h):
        c_multi.append(generate_contex(embeddings, d_q, d_k, d_v))
    
    context = torch.stack(c_multi, dim=0)
    print(f"Multi-head context: {context}")
    print(f"Shape of multi-head context: {context.shape}")
    return context

s = "Attention is all you need for now"

embeddings = generate_token_embeddings(s)
generate_contex(embeddings, 24, 24, 28)
generate_multi_contex(embeddings, 24, 24, 28, 5)

