import torch

@torch.no_grad()
def kernel(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    local_expert_offset: int,
    routed_scaling_factor: float,
    output: torch.Tensor, # 框架预分配好的 [T, H] bf16 张量
):
    """
    基于官方 Reference 修改的纯 PyTorch Baseline，用于跑通评测。
    """
    # 尺寸参数定义 (DeepSeek V3 / R1 强绑定)
    H = 7168
    I = 2048
    E_local = gemm1_weights.shape[0]
    
    BLOCK = 128
    E_global = routing_logits.shape[1]
    T = routing_logits.shape[0]

    TOP_K = 8
    N_GROUP = 8
    TOPK_GROUP = 4

    device = hidden_states.device

    # 1) FP8 block-scale Dequantization (极其复杂的反量化逻辑)
    # 激活值反量化
    A_fp32 = hidden_states.to(torch.float32)
    A_scale = hidden_states_scale.to(torch.float32)            
    A_scale_TH = A_scale.permute(1, 0).contiguous()            
    A_scale_expanded = (
        A_scale_TH.unsqueeze(-1)
        .repeat(1, 1, BLOCK)                                   
        .reshape(T, H)                                         
        .contiguous()
    )
    A = A_fp32 * A_scale_expanded                              

    # 权重 W13 反量化 (Gate 和 Up 合在一起了，叫 W13)
    W13_fp32 = gemm1_weights.to(torch.float32)
    S13 = gemm1_weights_scale.to(torch.float32)
    S13_expanded = torch.repeat_interleave(S13, BLOCK, dim=1)  
    S13_expanded = torch.repeat_interleave(S13_expanded, BLOCK, dim=2)  
    W13 = W13_fp32 * S13_expanded                              

    # 权重 W2 反量化 (Down Proj)
    W2_fp32 = gemm2_weights.to(torch.float32)
    S2 = gemm2_weights_scale.to(torch.float32)
    S2_expanded = torch.repeat_interleave(S2, BLOCK, dim=1)    
    S2_expanded = torch.repeat_interleave(S2_expanded, BLOCK, dim=2)    
    W2 = W2_fp32 * S2_expanded                                 

    # 2) DeepSeek 独占的 No-aux Routing (无辅助损失分组路由)
    logits = routing_logits.to(torch.float32)                      
    bias = routing_bias.to(torch.float32).reshape(-1)              

    s = 1.0 / (1.0 + torch.exp(-logits))                       
    s_with_bias = s + bias                                     

    group_size = E_global // N_GROUP 
    s_wb_grouped = s_with_bias.view(T, N_GROUP, group_size)    

    top2_vals, _ = torch.topk(s_wb_grouped, k=2, dim=2, largest=True, sorted=False)  
    group_scores = top2_vals.sum(dim=2)                        

    _, group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=False)  
    group_mask = torch.zeros_like(group_scores)                
    group_mask.scatter_(1, group_idx, 1.0)
    score_mask = group_mask.unsqueeze(2).expand(T, N_GROUP, group_size).reshape(T, E_global)  

    neg_inf = torch.finfo(torch.float32).min
    scores_pruned = s_with_bias.masked_fill(score_mask == 0, neg_inf)                  
    _, topk_idx = torch.topk(scores_pruned, k=TOP_K, dim=1, largest=True, sorted=False)  

    M = torch.zeros_like(s)                                    
    M.scatter_(1, topk_idx, 1.0)                               
    weights = s * M                                            
    weights_sum = weights.sum(dim=1, keepdim=True) + 1e-20
    weights = (weights / weights_sum) * routed_scaling_factor  

    # 3) Local expert compute and accumulation
    # 先新建一个临时的 f32 张量存累加结果
    temp_output = torch.zeros((T, H), dtype=torch.float32, device=device)
    local_start = int(local_expert_offset)

    for le in range(E_local):
        ge = local_start + le
        if ge < 0 or ge >= E_global:
            continue

        sel_mask_per_token = (topk_idx == ge).any(dim=1)       
        if not sel_mask_per_token.any():
            continue

        token_idx = torch.nonzero(sel_mask_per_token, as_tuple=False).squeeze(1)  
        Tk = token_idx.numel()

        A_e = A.index_select(0, token_idx)                     
        W13_e = W13[le]                                        
        W2_e = W2[le]                                          

        # GEMM1 -> SwiGLU -> GEMM2
        G1 = A_e.matmul(W13_e.t())                             
        X1 = G1[:, :I]                                         
        X2 = G1[:, I:]                                         
        silu_X2 = X2 / (1.0 + torch.exp(-X2))                  
        C = silu_X2 * X1                                       
        O = C.matmul(W2_e.t())                                 

        w_tok = weights.index_select(0, token_idx)[:, ge]      
        temp_output.index_add_(0, token_idx, O * w_tok.unsqueeze(1))  

    # 4) 将结果写回框架提供的 output 张量中（关键的一步，必须 inplace！）
    output.copy_(temp_output.to(output.dtype))