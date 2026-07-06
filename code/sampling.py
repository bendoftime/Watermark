import torch
from IPython import embed
from alternative_prf_schemes import prf_lookup

def seed_rng(generator, tokens, seeding_scheme="minhash_prf", hash_key=15485863, c=5):
    # Seed RNG from local context. Not batched, because the generators we use (like cuda.random) are not batched.
    # Borrowed from 
    # https://github.com/jwkirchenbauer/lm-watermarking/blob/main/watermark_reliability_release/watermark_processor.py
    # tokens should be in the shape of (1, current_length)

    assert tokens.shape[-1] >= c, f"seeding_scheme={seeding_scheme} requires at least a {c} token prefix sequence to seed rng"
    prf_key = prf_lookup[seeding_scheme](tokens[0][-c:], salt_key=hash_key)
    generator.manual_seed(prf_key)
    # 将上一步计算得到的伪随机密钥强制设定为生成器的种子。这保证了只要前 c 个词和密钥相同，接下来生成的随机数就完全一致。


## For Gumbel-max watermarks
def gumbel_key_func(generator,inputs,vocab_size, key, c, seeding_scheme):
    # add randonseed
    xis = []
    # 初始化列表，用于存储生成的伪随机数变量 xi（对应论文中的 U 变量）。
    pis = []
    # 初始化列表，用于存储序列排列。在 Gumbel-max 中排列不改变分布，所以这里其实是占位符，为了与后面的逆变换接口保持一致。
    for k in range(inputs.shape[0]):
    # 遍历批次（batch）中的每一条文本序列。
        seed_rng(generator, inputs[k].unsqueeze(0), seeding_scheme=seeding_scheme, hash_key=key, c=c) # This function require inputs of the shape (1, Length)
        # 提取第 k 条序列，扩展维度使其形状为 (1, Length)，然后调用前面的 seed_rng 函数为其独立设置随机数种子。
        xi = torch.rand(size=(1,vocab_size), generator=generator)
        # 生成大小为 [1, 词表大小] 的 0 到 1 之间的均匀分布随机数，存入 xi。这对应论文中给词表里每个词生成的一个独立均匀随机数。
        pi = torch.arange(vocab_size)
        # 生成一个从 0 到 vocab_size-1 的整数序列，作为词表索引的排列。这里实际上没有打乱顺序，因为 Gumbel-max 中排列不改变分布。
        xis.append(xi)
        pis.append(pi)
    xis=torch.vstack(xis)
    # 使用 vstack 将列表中的多个形状为 (1, vocab_size) 的张量在第 0 维度（batch 维度）拼接起来。
    pis=torch.vstack(pis)
    return xis,pis

def gumbel_sampling(probs,pi,xi):
# 定义 Gumbel-max 采样函数。执行论文中 w_t = argmax(U_{t,w}^{1/P_{t,w}}) 的水印生成逻辑。
    return torch.argmax(xi ** (1/torch.gather(probs, 1, pi)),axis=1).unsqueeze(-1)
    # 首先 torch.gather 根据排列 pi 重新排列概率 probs（这里 pi 是按序的，所以等同于 probs）。
    # 接着，取伪随机数 xi 的 (1/预测概率) 次方。最后，在词表维度 (axis=1) 上用 argmax 找出最大值所在的索引，即为要采样的目标词。扩展最后一个维度以适配输出。
    # 输出的s的形状为 [batch_size, 1]，每一行表示对应序列在词表中的采样索引。

def gumbel_Y(s, pi, xi):
# 定义计算 Gumbel-max 水印检测所用统计量（枢轴统计量）的函数。
    xi_samp = torch.gather(xi,-1,s.cpu()).squeeze(-1)
    #注意！这里将.squeeze()改为.squeeze(-1)，以确保只去掉最后一个维度，避免在 batch_size > 1 时出现问题。
    # 根据采样出来的目标词索引 s，使用 gather 从当步生成的随机数矩阵 xi 中提取该目标词所对应的那个伪随机数值 U_{t, w_t}。
    # xi: 代表在当前时间步，系统为词表中每一个候选词生成的U(0,1)标准均匀分布伪随机数向量。它的形状通常是 [batch_size, vocab_size]。
    # s: 代表经过采样算法（即 gumbel_sampling 函数）最终选定输出的词的索引。它的形状通常是 [batch_size, 1]。
    # 在 PyTorch 的底层实现中，torch.gather 要求被提取的数据张量（xi）和索引张量（s）必须处于同一个硬件设备上。
    # 由于前面生成伪随机数 xi 的操作和存储可能发生在 CPU 上，而模型推理和 s 的计算通常发生在 GPU 上，
    # 因此这里调用 .cpu() 是为了确保设备匹配，防止内存设备不一致导致的报错。
    # torch.gather(xi, -1, s.cpu()): -1 表示在张量的最后一个维度（即 vocab_size 维度，代表词表）上进行查找提取。
    # s.cpu() 提供了要提取的index（当前生成的词的 ID）。
    # 对于Batch中的第i个句子，它会在 xi 的第i行中，去寻找索引为 s[i][0] 的那个数值，并把它取出来。提取后得到的结果依然是一个二维张量，形状为 [batch_size, 1]。
    # 注意对于torch.gather, 它通过替换指定维度 dim 上的坐标，来决定输出张量中每个位置的值。输出张量的形状与 index 的形状完全一致。
    # 1. 维度数量必须相等：input 和 index 必须具有相同的维度数量。例如 input 是 2 维，index 也必须是 2 维。
    # 2. 维度大小必须匹配：除了指定的 dim 维度外，index 在其他所有维度上的大小，必须小于或等于 input 在对应维度上的大小。
    return xi_samp
    # 返回该值。在论文中，这个值就是 Y_t^{gum}，用来做后续的显著性检验。


## For inverse transform watermarks
def transform_key_func(generator,inputs,vocab_size, key, c, seeding_scheme):
    batch_size = inputs.shape[0] # batch_size must be 1
    assert batch_size == 1, "Batch size should be 1 for the inverse transform watermark!"
    # add randonseed
    xis = []
    pis = []
    for _ in range(batch_size):
        seed_rng(generator, inputs, seeding_scheme=seeding_scheme, hash_key=key, c=c)
        xi = torch.rand(size=(batch_size,1), generator=generator)
        pi = torch.randperm(vocab_size, generator=generator)
        xis.append(xi)
        pis.append(pi)
    xis=torch.vstack(xis)
    pis=torch.vstack(pis)
    return xis,pis

def inv(perm):
    inverse = [0] * len(perm)
    for i, p in enumerate(perm):
        inverse[p] = i
    return inverse

def inverse_permutation(perm):
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.size(0), device=perm.device)
    return inv


def transform_sampling(probs,pi,xi):
    inv_pi = inverse_permutation(pi.squeeze()).unsqueeze(0)
    cdf = torch.cumsum(torch.gather(probs, 1, inv_pi), 1)
    s = torch.gather(inv_pi, 1, torch.searchsorted(cdf, xi))
    return s


def transform_Y(s, pi, xi):
    ## For dif: Y = -|U - eta|. 
    ## Unlike the form we introduced in our paper, we add a minus in experiments
    ## In this way, we will have E_0 Y < E_1 Y.
    vocab_size = pi.shape[1]
    s_samp = torch.gather(pi,-1,s.cpu()).squeeze() 
    return -torch.abs(xi-(s_samp-1)/(vocab_size-1)), xi, (s_samp-1)/(vocab_size-1)
