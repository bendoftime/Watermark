import torch


def generate_inv(model,prompts,vocab_size,m,key_func,sampler, Y_func, key=23333,c=5, seeding_scheme="minhash_prf",temperature=0.1):
    generator = torch.Generator()
    inputs = prompts.to(model.device)
    attn = torch.ones_like(inputs)
    past = None

    Ys = []
    Us = []
    etas = []
    top_probs = []
    for _ in range(m): 
        with torch.no_grad():
            if past:
                output = model(inputs[:,-1:], past_key_values=past, attention_mask=attn)
            else:
                output = model(inputs)

        probs = torch.nn.functional.softmax(output.logits[:,-1]/temperature, dim=-1).cpu()
        top_prob = torch.max(probs, axis=1)[0].unsqueeze(0)
        xi, pi = key_func(generator,inputs, vocab_size, key, c, seeding_scheme)  
        tokens = sampler(probs, pi, xi).to(model.device) 
        Y, U, eta = Y_func(tokens, pi, xi)

        inputs = torch.cat([inputs, tokens], dim=-1)

        Ys.append(Y.unsqueeze(0))
        Us.append(U.unsqueeze(0))
        etas.append(eta.unsqueeze(0))
        top_probs.append(top_prob)

        past = output.past_key_values
        attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)

    Ys = torch.vstack(Ys)
    Us = torch.vstack(Us)
    etas = torch.vstack(etas)
    top_probs = torch.vstack(top_probs)
    return inputs.detach().cpu(), Ys.detach().cpu(),Us.detach().cpu(), etas.detach().cpu(),top_probs.detach().cpu()


def generate_gum(model,prompts,vocab_size,m,key_func,sampler, Y_func, key=23333,c=5, seeding_scheme="minhash_prf",temperature=0.1):
    generator = torch.Generator()
    # 实例化一个独立的 PyTorch 随机数生成器对象，后续用于生成伪随机数。
    inputs = prompts.to(model.device)
    # 将输入的提示词张量移动到模型所在的计算设备（如 GPU）上。
    attn = torch.ones_like(inputs)
    # 创建一个与 inputs 形状相同的全 1 张量，作为注意力掩码（Attention Mask），告诉模型所有的输入词都是有效的。
    past = None
    # 为了后续使用 KV Cache（键值缓存）技术加速文本生成。

    Ys = []
    top_probs = []
    for _ in range(m): 
    # 逐个生成 m 个词 
        with torch.no_grad():
        # 在 PyTorch 中，默认情况下执行的所有张量（Tensor）运算都会被记录在一张“计算图”中，
        # 以便在训练阶段使用反向传播（Backpropagation）计算梯度并更新模型权重。
        # 然而，当前的阶段是推理（Inference/Generation），我们只需要模型正向输出结果，不需要更新权重。
        # with torch.no_grad(): 作为一个上下文管理器，强制 PyTorch 停止追踪和记录梯度信息。
        # 这可以释放掉原本用于存储前向传播中间激活值（Activations）的巨量显存（VRAM），
        # 并免去构建计算图的 CPU/GPU 开销，使生成速度大幅提升，显存占用大幅降低。
            if past:
            # KV Cache, 这是 Transformer 架构推理中最关键的优化技术。
            # 大模型的生成是“逐词生成”的（预测出第1个词，拼接到输入里，再去预测第2个词，依此类推）。
                output = model(inputs[:,-1:], past_key_values=past, attention_mask=attn)
                # Note [:,-1:] not [:,-1]. We need 2D.
                # 当生成第二个词时，如果没有 KV Cache，模型需要重新计算 101 个词的注意力，这会导致严重的重复计算（复杂度呈平方级增长）。
                # 使用了 KV Cache 后，代码执行 output = model(inputs[:, -1:], past_key_values=past, ...)。
                # 注意 inputs[:, -1:] 的切片操作：它表示只截取序列的最后一个词（即上一轮刚刚生成的新词）输入给模型。
                # 模型拿到这个单词后，只需计算这一个词的 Query 向量，然后将其与之前缓存的（past）历史 Key 和 Value 矩阵进行注意力点积运算。
                # 这使得每一步的时间复杂度被压缩到了O(1)（相对于序列长度），极大地加速了生成过程。
            else:
                output = model(inputs)
                # 当生成第一个词时，past 为 None。此时代码执行 output = model(inputs)。
                # 模型读取完整的初始提示词序列（例如长度为 100），
                # 计算所有 100 个词的 Query、Key、Value 向量，执行完整的自注意力（Self-Attention）计算，
                # 输出结果，并将这 100 个词的 Key 和 Value 矩阵保存在 output.past_key_values 中。

        probs = torch.nn.functional.softmax(output.logits[:,-1]/temperature, dim=-1).cpu()
        # output.logits：模型最后一层的输出。它是一个形状为 (Batch_Size, Sequence_Length, Vocab_Size) 的三维张量。
        # 里面装的是未归一化的原始得分（Logits），数值可以是正数也可以是负数。
        # [:, -1]：在序列长度（Sequence_Length）这个维度上，取索引为 -1 的元素，即最后一个时间步的输出。
        # 因为我们只关心序列末尾“下一个即将出现的词”的预测得分，前面词的预测得分在当前步骤毫无意义。提取后，形状变为 (Batch_Size, Vocab_Size)。
        # / temperature：将所有 Logits 除以一个标量 temperature（代码中默认设为 0.1）。这是一个统计热力学借鉴来的概念：
        # 当T=1时，对原始分布不作改变。
        # 当T<1时，除以 0.1 等同于将 Logits 放大 10 倍。这会使得原本得分高的词优势被极度放大，得分低的词被极度缩小。
        # 后续经过 Softmax 后，概率分布会变得极其尖锐，模型输出会变得非常确定和稳定。
        # 当T>1时，会使分布变得平缓，增加输出的随机性和多样性。
        # softmax(..., dim=-1)：应用 Softmax 函数 e^{x_i}/sum_j e^{x_j}, 
        # 将对最后一个维度（词表维度）的原始得分强制转换为概率分布。所有的值都被压缩到[0,1],且加和严格等于 1。
        # .cpu()：将计算好的概率矩阵从显存（GPU）拉取回主存（CPU），以便后续（可能的）基于 CPU 的随机数采样函数使用。
        top_prob = torch.max(probs, axis=1)[0].unsqueeze(0)
        # torch.max(probs, axis=1)：在词表维度（axis=1）上寻找最大值。该函数会返回两个张量：
        # 第一个是最大值本身（即置信度最高的词的概率），第二个是最大值所在的索引（即该词在词表中的 ID）。
        # [0]：取返回结果的元组中的第一个元素，也就是只提取最大概率的具体数值，丢弃索引信息。
        # .unsqueeze(0)：在第 0 维度增加一个尺寸为 1 的新维度。
        # 这一步是张量维度的对齐操作，目的是为了后续能够使用 torch.vstack 将所有时间步的最高概率顺畅地垂直堆叠成一个完整的矩阵。
        xi, pi = key_func(generator,inputs, vocab_size, key, c, seeding_scheme) 
        # 伪随机数 
        tokens = sampler(probs, pi, xi).to(model.device) 
        # 带水印的新词
        Y = Y_func(tokens, pi, xi)
        # 记录pivotal quantitiy

        inputs = torch.cat([inputs, tokens], dim=-1)
        # 将生成的新词拼接到输入序列的末尾

        Ys.append(Y.unsqueeze(0))
        top_probs.append(top_prob)

        past = output.past_key_values
        # 提取当前的 KV Cache（对历史文本的理解），留给下一步使用，避免重复计算前文
        attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)
        # 在生成了一个新词之后，把模型的“注意力掩码（Attention Mask）”加长一格，告诉模型这个新词也是合法的，下一步计算时需要关注它。
        # attn.shape[0]：获取行数（句子数量）。attn 是一个二维表格（矩阵），形状是 (Batch_Size, Sequence_Length)，即 (句子数, 词数)。
        # attn.new_ones((attn.shape[0], 1))：量身定制一个新的全 1 小列
        # 我们需要为刚刚生成的新词加上注意力标记（标记为 1，代表有效词汇）。因此我们需要造一个全部由 1 组成的新矩阵。
        # (attn.shape[0], 1) 规定了新矩阵的形状：行数和原来的 attn 保持一致，列数是 1 列。
        # dim=-1“沿着列的方向（横向）”把它们拼起来。
        # 这里为什么用 attn.new_ones()，而不是最常见的 torch.ones()？
        # 因为如果直接用 torch.ones()，PyTorch 默认会在主内存（CPU）上生成这个矩阵，而且数据类型可能是普通的浮点数。
        # 但原来的 attn 矩阵此刻大概率是躺在显卡（GPU）里的。
        # 在深度学习中，CPU 的数据和 GPU 的数据是绝对不能直接拼接的，否则程序直接崩溃。
        # attn.new_ones() 的高级之处在于，
        # 它会“克隆”原变量 attn 的所有物理属性（在哪个 GPU 上、是什么数据格式），然后在完全相同的地方生成这个新矩阵。这就保证了绝对的安全和兼容。
    Ys = torch.vstack(Ys)
    # 循环m次结束后，torch.vstack将收集到的所有单个步骤的数据拼接成完整的矩阵
    top_probs = torch.vstack(top_probs)
    return inputs.detach().cpu(), Ys.detach().cpu(), top_probs.detach().cpu(), 
    # 切断它们与 PyTorch 计算图的联系（detach()），转移回系统主内存（cpu()），最后返回生成的完整文本 inputs，以及 Ys 和 top_probs。


# generate unwatermarked completions of token length m given list of prompts
def generate_rnd(prompts,m,model):
    inputs = prompts.to(model.device)
    attn = torch.ones_like(inputs)
    past = None
    for i in range(m):
        with torch.no_grad():
            if past:
                output = model(inputs[:,-1:], past_key_values=past, attention_mask=attn)
            else:
                output = model(inputs)

        probs = torch.nn.functional.softmax(output.logits[:,-1], dim=-1)
        
        tokens = torch.multinomial(probs,1)
        inputs = torch.cat([inputs, tokens], dim=1)

        past = output.past_key_values
        attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)
    
    return inputs.detach().cpu()
