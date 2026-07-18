from time import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
from collections import defaultdict
import pickle
import copy
import numpy as np
from generation import generate_inv, generate_gum, generate_rnd
from sampling import transform_sampling, transform_key_func, transform_Y
from sampling import gumbel_sampling, gumbel_key_func, gumbel_Y
import argparse

results = defaultdict(dict)
## We only generate the data.

parser = argparse.ArgumentParser(description="Experiment Settings")

parser.add_argument('--method',default="gumbel",type=str)
# parser.add_argument('--method',default="transform",type=str)
# parser.add_argument('--method',default="raw",type=str)

parser.add_argument('--model',default="facebook/opt-1.3b",type=str)
# parser.add_argument('--model',default="facebook/opt-2.7b",type=str)
# parser.add_argument('--model',default="princeton-nlp/Sheared-LLaMA-2.7B",type=str)
# parser.add_argument('--model',default="huggyllama/llama-7b",type=str)

parser.add_argument('--seed',default=15485863,type=int)
parser.add_argument('--c',default=4,type=int)
parser.add_argument('--temp',default=0.1,type=float)

parser.add_argument('--batch_size',default=1,type=int)
parser.add_argument('--seed_way',default="skipgram_prf",type=str)
parser.add_argument('--m',default=200,type=int)
parser.add_argument('--T',default=500,type=int)

parser.add_argument('--prompt_tokens',default=50,type=int)
parser.add_argument('--buffer_tokens',default=20,type=int)
parser.add_argument('--max_seed',default=100000,type=int)

parser.add_argument('--norm',default=1,type=int)
parser.add_argument('--truncate_vocab',default=8,type=int)

args = parser.parse_args()
results['args'] = copy.deepcopy(args)
print(args)

# fix the random seed for reproducibility
t0 = time()
# 记录当前时间，用于后续计算模型加载花了多少秒。
torch.manual_seed(args.seed)
# 设置 PyTorch 的全局随机种子，确保后续在相同的设置下能够生成近乎一模一样的结果。
# 但是注意：NumPy 的蒙特卡洛阈值没有设种子，GPU 算子、依赖版本和流式数据顺序也可能影响结果。
# 所以并不是完全一模一样的结果。
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 检查当前环境是否有可用的 Nvidia GPU，如果有就使用第一块 GPU ("cuda:0")，否则退回到 CPU 计算。

tokenizer = AutoTokenizer.from_pretrained(args.model)
# 根据传入的 `--model` 参数，从 Hugging Face 联网下载或从本地缓存加载对应的分词器 (Tokenizer)。
model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
# 加载对应的因果语言模型权重，并将其整体转移到之前指定的硬件设备（GPU或CPU）上。

## Label the used model
if args.model == "facebook/opt-1.3b":
    model_name = "1p3B"
elif args.model == "huggyllama/llama-7b":
    model_name = "7B"
elif args.model == "princeton-nlp/Sheared-LLaMA-2.7B":
    model_name = "2p7B"
else: 
    raise ValueError(f"No such a model: {args.model}.")

vocab_size = model.get_output_embeddings().weight.shape[0]
# 获取模型输出层（通常是最后的线性层，将隐状态映射回词汇表概率）的权重矩阵的第一维大小，也就是模型的词汇表容量 (vocab size)。
eff_vocab_size = vocab_size - args.truncate_vocab
# 代码里实际没有用到eff_vocab_size
# 计算有效词汇表大小，即总词汇表大小减去通过参数设定要截断掉的数量。
# truncate_vocab（词汇表截断）是一个工程实现上的安全保护机制。
# 它的核心作用是：屏蔽掉模型词汇表末尾那几个“仅仅为了凑数而存在、毫无实际意义的幽灵单词”，防止加水印时发生崩溃。
# 在训练大语言模型时，为了让 GPU（显卡）内部的张量核心（Tensor Cores）发挥出极限速度，矩阵的维度最好是 8、16 或 64 的整数倍。
# 以代码中默认使用的 OPT-1.3B 模型为例：它真实认识的有用单词（Token）数量实际上只有50265 个。
# 但这个数字不是 8 或 16 的倍数，算起来不够快。
# 于是，开发者会在词汇表的最后，强行塞入 7 个毫无意义的“空占位符（Padding Tokens）”，把总词汇表强行扩充到 50272（这是一个能被 64 完美整除的漂亮数字）。
# （作者在代码中为了保守起见，设置了 --truncate_vocab 8，多切掉一个冷僻字符完全不影响大局，但能绝对保证安全）。
# 这些词没有任何现实语义，分词器（Tokenizer）根本不认识它们，它们仅仅是物理内存里的“占位石头”。

# 自动计算差值。如果差值小于 0，则设为 0。
# auto_truncate = max(0, vocab_size - len(tokenizer))
# eff_vocab_size = vocab_size - auto_truncate
print(f'Loaded the model (t = {time()-t0} seconds)')
print()
print("The vocabulary size is", vocab_size)
print()
dataset = load_dataset("allenai/c4", "realnewslike", split="train", streaming=True)
# 加载 C4 数据集中的 "realnewslike"（类真实新闻）子集，使用 "train" 拆分部分。`streaming=True` 代表以流式读取，不一次性把庞大的数据集全下载到内存中，节约空间。
# 这段代码不是在训练模型。 代码的目的只是为了从网上找一些人类写的新闻开头，喂给模型让它续写并打上水印。既然不训练模型，为什么要指定拿“训练集”呢？
# 原因很简单：为了拿到尽可能多的文本。Train split体量最大（通常占 80% 甚至更多）。

T = args.T                                    # number of prompts/generations
n_batches = int(np.ceil(T / args.batch_size)) # number of batches
prompt_tokens = args.prompt_tokens            # minimum prompt length
new_tokens = args.m                           # number of tokens to generate
buffer_tokens = args.buffer_tokens 
# buffer_tokens 的核心作用是为了应对文本在“解码再重新分词（Retokenization）”时造成的长度缩水问题，而特意设置的“冗余/安全余量”。
# 在代码和实验设计中，它的作用体现在应对 BPE 分词算法的“合并缩水”效应
# 这是论文作者在附录 D.2 中专门提到的一个技术细节。
# （但与这份代码的实际行为不一致。代码没有解码、重新分词或 padding；它只是为 null 样本生成 220 个 token，而检测只使用前 200 个，因此额外 20 个 token 实际被丢弃。）
# 大模型的 Token 与人类阅读的单词并不是绝对的 1:1 映射。分词器（Tokenizer）使用的是子词（Subword）机制。
# 生成阶段：大模型可能生成了两个独立的 Token，例如 [" water", "melon"]。当实验把生成的 Token ID 转换回人类可读的字符串并保存到硬盘时，它变成了 " watermelon"。
# 检测阶段：当后续运行水印检测脚本时，检测程序需要把字符串 " watermelon" 重新输入给分词器转换成 Token。
# 此时，分词器会发现这是一个完整的常用词，直接将其编码为一个单一的 Token [" watermelon"]。
# 后果：由于这种“碎片合并”现象的存在，如果在生成阶段刚好要求模型生成 200 个 Token，等存成文本再读取检测时，这段文本的长度通常会缩水到 195 甚至更少。
# 但水印的统计学检验公式（如计算第一/第二类错误率）要求检测样本的 Token 长度必须严格达到预设的数值 n。
# 如果长度不够，就必须强行在末尾填充（Padding）无意义的特殊字符来凑数，这会严重干扰统计数据的纯净度。
# 作者引入了 buffer_tokens (20 by default)。
# 在生成对照组文本时（如代码中的 generate_rnd 函数），代码会要求模型生成 new_tokens + buffer_tokens（即 200 + 20 = 220 个 Token）。
# 多生成这 20 个“缓冲词”，确保了即便在保存和重新读取时发生了分词合并损耗，剩余的真实文本长度也绝对能大于 200，从而完全避免了使用人工 Padding 凑数。


if args.method == "transform":
    generate_null = False
    generate_watermark = lambda prompt : generate_inv(model,
                                                  prompt,
                                                  vocab_size,
                                                  new_tokens,
                                                  transform_key_func,
                                                  transform_sampling,
                                                  transform_Y,
                                                  key=args.seed,
                                                  c=args.c,
                                                  seeding_scheme=args.seed_way,
                                                  temperature=args.temp)

elif args.method == "gumbel":
    generate_null = False
    generate_watermark = lambda prompt : generate_gum(model,
                                                  prompt,
                                                  vocab_size,
                                                  new_tokens,
                                                  gumbel_key_func,
                                                  gumbel_sampling,
                                                  gumbel_Y,
                                                  key=args.seed,
                                                  c=args.c,
                                                  seeding_scheme=args.seed_way,
                                                  temperature=args.temp)
elif args.method == "raw":
    generate_null = True                                                                  
else:
    raise

ds_iterator = iter(dataset)
# 将流式加载的数据集转换为一个可迭代对象，方便逐条取数据。

t1 = time()
# 记录开始提取提示词和生成文本的时间。

## Get T=500 prompts which length is truncated to m=200
# （这里有点小问题，应该是 truncated to prompt_tokens=50）
prompts = []
# 初始化一个空列表，用来装提取出来的提示词。
itm = 0
# 设立一个计数器，记录目前已经成功提取了多少条可用的提示词。
while itm < T:
# 开始循环，只要拿到手的提示词数量不够 T ，就一直抽取。
    example = next(ds_iterator)
    # 从数据集迭代器中获取下一篇文章。
    text = example['text']
    # 提取这篇数据集文章里的具体文本内容。

    tokens = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=2048-buffer_tokens)[0]
    # 调用大语言模型的Tokenizer，将输入的原始新闻文本 text 切割成Token，并将每个词元映射为模型词汇表中对应的整数 ID（例如把 "apple" 转换为 1523）。
    # return_tensors='pt' 表示返回的结果是 PyTorch 张量格式，这是因为后续输入给深度学习模型的数据必须是 PyTorch 张量格式。
    # truncation=True 表示如果文本长度超过了指定的最大长度 max_length，就会自动截断，确保不会超出模型的输入限制。
    # max_length=2048-buffer_tokens 将候选 C4 文档最多截断到 2048-buffer_tokens 个 token。
    # 注意：后续实际只提取 prompt_tokens 个 token 作为模型输入，因此在当前实现中，这并不是上下文长度保护。
    # 而是决定从文档的哪个位置截取。这里默认右截断，对于长文本配合prompt_tokens的选取方式，更容易能截取到正文部分。
    if len(tokens) < prompt_tokens + new_tokens:
    # 如果当前文章的 Token 数量不足以提供至少 prompt_tokens 个提示词和 new_tokens 个生成词元，就直接跳过这篇文章，继续抽取下一篇。
        continue
    prompt = tokens[-(new_tokens+prompt_tokens):-new_tokens]
    # 在当前文章末尾预留出 `new_tokens` 长度的空间，截取长度为 `prompt_tokens` 的片段作为大模型的 prompt 输入。
    prompts.append(prompt)

    itm += 1
prompts = torch.vstack(prompts)
# 循环结束后，将包含 T 个一维张量的列表，vstack成一个形状为 (T, prompt_tokens) 的二维张量矩阵，方便批处理计算。
results['prompts'] = copy.deepcopy(prompts)
# 深拷贝并保存这些提示词张量到结果字典中，以备实验比对时重现输入使用。
# 在代码中使用 copy.deepcopy()，核心原因是出于Defensive Programming的考虑。
# 它的目的是：彻底切断变量之间的内存引用关联，把当前这一刻的数据安全地“封存”进结果字典中，防止后续代码的任何操作意外篡改已经保存好的数据。

if not generate_null:
    ## If we need to generate watermarked samples

    if args.method == "transform":
        watermarked_samples = []
        generated_Ys = []
        generated_Us = []
        generated_etas = []
        generated_top_probs = []
        for batch in tqdm(range(n_batches)):
            idx = torch.arange(batch * args.batch_size,min(T,(batch + 1) * args.batch_size))

            generated_tokens, Ys, Us, etas, top_probs = generate_watermark(prompts[idx])
            watermarked_samples.append(generated_tokens[:,prompt_tokens:])
            generated_Ys.append(Ys.squeeze())
            generated_Us.append(Us.squeeze())
            generated_etas.append(etas.squeeze())
            generated_top_probs.append(top_probs.squeeze())

        watermarked_samples = torch.vstack(watermarked_samples)
        generated_Ys = torch.vstack(generated_Ys)
        generated_Us = torch.vstack(generated_Us)
        generated_etas = torch.vstack(generated_etas)
        generated_top_probs = torch.vstack(generated_top_probs)

        ## Save generated texts and pivotal statsitics
        results['watermark']['tokens'] = copy.deepcopy(watermarked_samples)
        results['watermark']['Ys'] = copy.deepcopy(generated_Ys)
        results['watermark']['Us'] = copy.deepcopy(generated_Us)
        results['watermark']['etas'] = copy.deepcopy(generated_etas)
        results['watermark']['top_probs'] = copy.deepcopy(generated_top_probs)

    elif args.method == "gumbel":
        watermarked_samples = []
        generated_Ys = []
        generated_top_probs = []
        for batch in tqdm(range(n_batches)):
        # 按批次进行带进度条的循环处理。
            idx = torch.arange(batch * args.batch_size,min(T,(batch + 1) * args.batch_size))
            # 根据批次索引和大小，计算当前处理的数据在提示词张量里对应的行号，利用 min 函数防止最后一批越界。

            generated_tokens, Ys, top_probs = generate_watermark(prompts[idx])
            # 取出对应行的 prompt，传入之前包装好的匿名函数进行生成。拿回生成的序列本身，以及伴随生成的重要统计量。
            watermarked_samples.append(generated_tokens[:,prompt_tokens:])
            # 把返回的完整序列里，原有的提示词部分切掉（[:,prompt_tokens:] 表示取行所有数据，列只取生成的部分），只保存模型新生成的词，然后追加到列表中。
            
            # 原代码：
            # generated_Ys.append(Ys.squeeze())
            # generated_top_probs.append(top_probs.squeeze())
            # 分别去掉这些统计量张量里多余的一维，加入对应的列表中。例如如果 Ys 的形状是 [200,1]，squeeze 后就变成 [200]。
            # 注意！这里原代码有问题，只适用于batch_size=1的情况。下面是修改后的版本：
            generated_Ys.append(Ys.transpose(0, 1))
            generated_top_probs.append(top_probs.transpose(0, 1))

        ## Save generated texts and pivotal statsitics
        watermarked_samples = torch.vstack(watermarked_samples)
        generated_Ys = torch.vstack(generated_Ys)
        generated_top_probs = torch.vstack(generated_top_probs)
        # 把列表里的每个小张量vstack成一个大张量，方便后续保存和分析。

        results['watermark']['tokens'] = copy.deepcopy(watermarked_samples)
        results['watermark']['Ys'] = copy.deepcopy(generated_Ys)
        results['watermark']['top_probs'] = copy.deepcopy(generated_top_probs)
    else:
        raise ValueError(f"This watermark method is not implemented: {args.method}.")
    
    print(f'Generated watermarked samples in (t = {time()-t1} seconds)')

    ## Name the experiment with configuration
    exp_name = f"{model_name}-{args.method}-c{args.c}-m{args.m}-T{args.T}-{args.seed_way}-{args.seed}-temp{args.temp}.pkl"
    pickle.dump(results,open(exp_name,"wb"))

else:
    ## If we need to generate unwatermarked samples 
    ## We don't adjust the temperature parameter here
    null_samples = []
    for batch in tqdm(range(n_batches)):
        idx = torch.arange(batch * args.batch_size,min(T,(batch + 1) * args.batch_size))
        null_samples.append(generate_rnd(prompts[idx],new_tokens+buffer_tokens,model)[:,prompt_tokens:])

    null_samples = torch.vstack(null_samples)
    results['null']['tokens'] = copy.deepcopy(null_samples)

    print(f'Generated samples in (t = {time()-t1} seconds)')

    ## Name the experiment with configuration
    exp_name = f"{model_name}-raw-m{args.m}-T{args.T}.pkl"
    pickle.dump(results,open(exp_name,"wb"))
