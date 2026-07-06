import numpy as np
from scipy.stats import gamma, norm, binom
from scipy.integrate import quad


def compute_gamma_q(q, check_point):
# 计算gamma分布quantile的函数。Aaronson 规则和对数规则在null hypothesis下统计量服从或可转化为gamma分布。
# Aaronson：h_{ars}(Y_t) = -\log(1 - Y_t)，用CDF method得知h_{ars}(Y_t)~ Exp(1)
# 文本序列长度为n时，总检验统计量T为n个单步得分的累加和，by iid，有Gamma(n,1)分布。
# 对数规则：h_{log}(Y_t) = log(Y_t)，用CDF method得知-log(Y_t)~ Exp(1)，所以-T follows Gamma(n,1)。
    qs = []
    for t in check_point:
    # check point对应序列长度
        qs.append(gamma.ppf(q=q,a=t))
        # 计算Gamma分布的quantile，其中a=t表示形状参数为t，q为分位数水平。
    return np.array(qs)


def compute_ind_q(q, ind_delta, check_point):
# 定义Indicator function规则通过CLT计算阈值的函数。
# h_{ind,delta}(r) = \mathbf{1}_{r >= delta}
    qs = []
    q = norm.ppf(q)
    for t in check_point:
        qs.append(t*(1-ind_delta)+ q*np.sqrt(t*(1-ind_delta)*ind_delta))
        # \mathbf{1}_{r >= delta}是一个Bernoulli分布，p=1-ind_delta，所以期望为1-ind_delta，方差为(1-ind_delta)*ind_delta。
    return np.array(qs)


def compute_general_q(q, mu, var, check_point):
# 定义通用的CLT阈值计算函数，适用于已知单次步骤理论均值 mu 和方差 var 的任意得分函数。
    qs = []
    q = norm.ppf(q)
    for t in check_point:
        qs.append(t*mu+ q*np.sqrt(t*var))
    return np.array(qs)


def compute_ind_q_new(q, mu, var, check_point):
# 这个函数应该是没有删掉
    qs = []
    q = norm.ppf(q)
    for t in check_point:
        qs.append(t*mu+ q*np.sqrt(t*var))
    return np.array(qs)


####################################################
##
## Compute test statistics for Gumbel-max watermarks
##
####################################################


def h_ars(Ys,  alpha=0.05):
# Aaronson 得分函数：h(y) = -log(1-y)。
    # Compute critical values
    check_points = np.arange(1, 1+Ys.shape[-1])
    # 提取输入数据 Ys 的列数（即文本最大长度），并生成从 1 到该长度的等差数组作为check point。
    h_ars_qs = compute_gamma_q(1-alpha, check_points)
    # 之前解释过了，用精确的gamma quantile计算阈值。

    # Compute the test scores
    Ys = np.array(Ys)
    h_ars_Ys = -np.log(1-Ys)
    cumsum_Ys = np.cumsum(h_ars_Ys, axis=1)

    results = (cumsum_Ys >= h_ars_qs)
    return np.mean(results,axis=0), np.std(results,axis=0)


def h_log(Ys,  alpha=0.05):
# log 得分函数：h(y) = log(y)。
    # Compute critical values
    check_points = np.arange(1, 1+Ys.shape[-1])
    h_log_qs = compute_gamma_q(alpha, check_points)
    # 因为有水印时-sum(log(Y_t))变得异常小（和aaronson相反），所以我们要看的是 Gamma 分布的左尾，所以这里是alpha而不是1-alpha。

    # Compute the test scores
    Ys = np.array(Ys)
    h_log_Ys = np.log(Ys)
    cumsum_Ys = np.cumsum(h_log_Ys, axis=1)

    results = (cumsum_Ys >= -h_log_qs)
    #-T<= h_log_qs iff T >= -h_log_qs
    return np.mean(results,axis=0), np.std(results,axis=0)


def h_ind(Ys, ind_delta=0.5, alpha=0.05):
# Indicator function 得分函数：h(y) = 1_{y>=delta}。
    # Compute critical values
    check_points = np.arange(1, 1+Ys.shape[-1])
    h_ind_qs = binom.ppf(n=check_points, p = 1-ind_delta, q = 1-alpha)
    # sum of Bernoulli(1-ind_delta) is Binomial(n, 1-ind_delta)

    # Compute the test scores
    Ys = np.array(Ys)
    h_ind_Ys = (Ys >= ind_delta)
    cumsum_Ys = np.cumsum(h_ind_Ys, axis=1)
    
    results = (cumsum_Ys >= h_ind_qs)
    return np.mean(results,axis=0), np.std(results,axis=0)


def h_opt_gum(Ys, delta0=0.2,theo=True, alpha=0.05):
# 实现论文定理 3.2 针对 Gumbel-max 推导出的理论最优得分函数。
    # Compute critical values
    Ys = np.array(Ys)
    check_points = np.arange(1, 1+Ys.shape[-1])

    def f(r, delta):
    # 根据theorem 3.2 定义最优得分函数。
        inte_here = np.floor(1/(1-delta))
        rest = 1-(1-delta)*inte_here
        return np.log(inte_here*r**(delta/(1-delta))+ r**(1/rest-1))
    h_ars_Ys = f(Ys, delta0)
    
    if theo:
    # 如果用理论推导法，用CLT计算阈值。
        mu = quad(lambda x: f(x, delta0), 0, 1,epsabs = 1e-10,epsrel=1e-10)
        # 使用数值积分求出该公式在null hypothesis U(0,1)下的理论期望。
        # quad: 这是 scipy.integrate 模块提供的数值积分函数，底层采用自适应的高斯积分算法。它可以高精度地计算一维定积分。
        # lambda x: f(x, delta0): 这是积分的目标函数（被积函数）。
        # 由于 quad 默认对传入函数的第一个参数进行积分，这里使用匿名函数 lambda 将 delta0 作为已知常数固定，使积分变量仅为 x。
        # 0, 1: 积分的下限和上限。
        # epsabs=1e-10: 设置积分的绝对误差容忍度，让数值积分算法在绝对误差小于该值时停止计算。
        # epsrel=1e-10: 设置积分的相对误差容忍度，让数值积分算法在相对误差小于该值时停止计算。
        EX2 = quad(lambda x: f(x, delta0)**2, 0, 1,epsabs = 1e-10,epsrel=1e-10)
        # 计算在null hypothesis U(0,1)下的E[X^2]，用于后续计算方差。

        mu, EX2 = mu[0], EX2[0]
        Var = EX2 - mu**2

        h_help_qs = compute_general_q(1-alpha, mu, Var, check_points)
        # 根据均值和方差，通过CLT求得最终的理论阈值。
    else:
    # 如果用数值模拟法，用蒙特卡洛模拟计算阈值。
        def find_q(N=500):
        # 定义一个闭包函数执行单轮模拟。
            Null_Ys = np.random.uniform(size=(N, Ys.shape[1]))
            # 生成服从均匀分布（null hypothesis，无水印）的随机样本。
            Simu_Y = f(Null_Ys, delta0)
            # 用公式对模拟无水印样本进行打分。
            Simu_Y = np.cumsum(Simu_Y, axis=1)
            # 对打分进行沿时间步的累加。
            h_help_qs = np.quantile(Simu_Y, 1-alpha, axis=0)
            # 直接在样本分布上取quantile。
            return h_help_qs
        
        q_lst = []
        for N in [500] * 10:
        # 循环执行 10 次独立模拟，注意每次模拟的样本量要和find_q函数中N参数一致。
            q_lst.append(find_q(N))
        h_help_qs = np.mean(np.array(q_lst),axis=0)
        # 对 10 次模拟算出的经验阈值求平均，以降低蒙特卡洛采样的方差

    cumsum_Ys = np.cumsum(h_ars_Ys, axis=1)
    results = (cumsum_Ys >= h_help_qs)
    return np.mean(results,axis=0), np.std(results,axis=0)


####################################################
##
## Compute test statistics for inverse watermarks
##
####################################################


def h_id_dif(Ds, alpha=0.05):
    # Compute critical values
    Ds = np.array(Ds)
    check_points = np.arange(1, 1+Ds.shape[-1])

    mu_dif = -1/3
    var_dif = 1/6 - 1/9
    h_id_dif_qs = compute_general_q(1-alpha, mu_dif, var_dif, check_points)

    # Compute the test scores
    cumsum_Ds = np.cumsum(Ds, axis=1)

    results = (cumsum_Ds >= h_id_dif_qs)
    return np.mean(results,axis=0), np.std(results,axis=0)


def h_ind_dif(Ds,delta=-0.05,alpha=0.05):
    assert -1 <= delta <= 0

    # Compute critical values
    Ds = np.array(Ds)
    check_points = np.arange(1,1+Ds.shape[1])
    h_id_dif_qs = binom.ppf(n=check_points, p = 1-(1+delta)**2, q = 1-alpha)

    # Compute the test scores
    h_ind_Ds = Ds >= delta
    cumsum_Ds = np.cumsum(h_ind_Ds, axis=1)

    results = (cumsum_Ds >= h_id_dif_qs)
    return np.mean(results,axis=0), np.std(results,axis=0)


def h_opt_dif(dif,delta=0.1,theo=False, weight=1e-6, vocab_size=32000, alpha=0.05, model_name="2p7B"):
    dif = np.array(dif)
    final_Y = dif

    def transform(Y):
        ## weight=1e-6 is used to avoid numerical blow-up
        return np.log(np.maximum(1+Y/(1-delta),weight)/np.maximum(1+Y,0)/(1-delta))
    
    # final_Y = np.log(np.maximum(1+final_Y/(1-delta),0)/np.maximum(1+final_Y,0))
    final_Y = transform(final_Y)
    cumsum_Ys = np.cumsum(final_Y, axis=1)

    # Compute critical values
    if theo:
        ## Thereotical critical values don't perform well due to the added truncation.
        def log_likelihood(x):
            return np.log(np.maximum(1+x/(1-delta), weight)/(1-delta)/(1+x))
        
        def rho(x):
            return 2*(1+x)
        
        mu =  quad(lambda x: rho(x)*log_likelihood(x), -1+1e-6, 0)[0]
        V2 = quad(lambda x: rho(x)*log_likelihood(x)**2, -1+1e-6, 0)[0]
        # print("mean", mu, "Var", V2-mu**2)

        h_opt_qs = compute_ind_q(1-alpha, mu, V2-mu**2, dif.shape[1])
    else:
        ## We use simulation to compute the critical values
        def find_q(N=1000):
            Null_Ys_U = np.random.uniform(size=(N, dif.shape[1]))
            Null_Ys_pi_s = np.random.randint(low=0, high=vocab_size, size=(N, dif.shape[1]))
            Null_etas = np.array(Null_Ys_pi_s)/(vocab_size-1)
            null_final_Y = -np.abs(Null_Ys_U-Null_etas)
            null_final_Y = transform(null_final_Y)
            null_cumsum_Ys = np.cumsum(null_final_Y, axis=1)
            h_opt_qs = np.quantile(null_cumsum_Ys, 1-alpha, axis=0)
            return h_opt_qs

        q_lst = []
        if model_name == "2p7B":
            ## If we use the 2.7B model, we should pay more efforts to control the Type I error
            for N in [500] * 10 + [200, 1000, 2000]: 
                q_lst.append(find_q(N))
            h_opt_qs = np.min(np.array(q_lst),axis=0)
        else:
            for N in [500] * 10: 
                q_lst.append(find_q(N))
            h_opt_qs = np.mean(np.array(q_lst),axis=0)

    results = (cumsum_Ys >= h_opt_qs)
    return np.mean(results,axis=0), np.std(results,axis=0)
