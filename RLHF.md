# RLHF

# 前言

​	

# 1.强化学习基础

## 1.1 有限马尔可夫决策过程



## 1.2价值函数与优势函数

​	**状态价值函数**，在Sutton的著作中，将状态价值函数的定义如下：
$$
v_{\pi}(s) \doteq \mathbb E_{\pi}\bigg[ G_t|S_t=s\bigg]=\mathbb E_{\pi}\bigg[ \sum_{k=0}^{\infin} \gamma^{k}R_{t+k+1} \bigg],\forall s \in \mathcal S
$$
​	状态价值函数反映了从状态 $S_t=s$ 开始，按照策略 $\pi$​执行动作后，智能体预计将获得的总奖励的期望值。



​	而价值函数满足某种递归关系，对于任何策略$\pi$和状态$s$，$s$的价值与其后继状态的价值关系如下：
$$
\begin{aligned} v_{\pi}(s)&\doteq \mathbb E_{\pi}\bigg[ G_t|S_t=s\bigg]\\
&=\mathbb E_{\pi}\bigg[ R_t+\gamma G_{t+1}|S_t=s\bigg] \\
&=\sum_{a}\pi(a|s)\sum_{r,s'} p(r,s'|s,a)(r+\gamma v_{\pi}(s'))\end{aligned}
$$
l	**动作价值函数** 则是根据策略$\pi$，从状态$s$开始，执行动作$a$之后所有可能的决策序列的期望回报，我们用符号$q_{\pi}(s,a)$表示：
$$
\begin{aligned}q_{\pi}(s,a)&\doteq  \mathbb E_{\pi}\bigg[ G_t|S_t=s,A_t=a\bigg]\\
&=\sum_{r,s'}p(r,s'|s,a)(r+\gamma v_{\pi}(s'))

\end{aligned}
$$
​	值得注意的是回报$R_t$是一个简写，完整的表示应该是$R(S_t,A_t,S_{t+1})$。我们将智能体在策略$\pi$控制下的预期折扣收益用符号$J(\pi)$表示：
$$
J(\pi)=\mathbb E_{\pi}\bigg[ G_t\bigg]=\mathbb E_{\pi}\bigg[ \sum_{t=0}^{\infin}\gamma^t R(S_t,A_t,S_{t+1}) \bigg]
$$
​	**公式变体**，在某些强化学习论文中，作者会为了简化公式的表达或者让某个特定的推导更清晰，而对标准符号做一些变动。如省略某些下角标或者是加上下角标变量，亦或者是对公式进行拓展，使得读者在阅读不同论文或材料时感到困惑，在本文接下来的内容中会涉及到不同的论文公式推导，为方便读者理解，笔者先列举一些公式的变体：
$$
\begin{aligned}J(\pi)&=\mathbb E_{\pi}\bigg[ \sum_{t=0}^{\infin}\gamma^t R(S_t,A_t,S_{t+1}) \bigg] \\
&= \mathbb E_{S_0,A_0,...}\bigg[ \sum_{t=0}^{\infin}\gamma^t R(S_t,A_t,S_{t+1}) \bigg]\\
&=\mathbb E_{\tau\sim(\pi,E)} \bigg[ G(\tau)\bigg] \\
&=\sum_{\tau \sim (\pi,E)} P(\tau|\pi)G(\tau)\end{aligned}
$$
​	公式的改写本质上是对同一个问题的不同表示方式，如上式，第一二行通常用于标准的动态规划方法或分析中，直接表示在时间步$t$处从状态$S_t$开始采取动作$A_t$所获得的累积奖励的期望。第三四行则是从策略 $\pi$和环境$E$生成的轨迹序$\tau$的角度出发。同样，状态价值函数与动作价值函数也可以进行适当改写：
$$
\begin{aligned} v_{\pi}(s)&\doteq \mathbb E_{\pi}\bigg[ G_t|S_t=s\bigg]\\
&=\mathbb E_{\pi}\bigg[ R_t+\gamma G_{t+1}|S_t=s\bigg] \\
&=\mathbb E_{S_t=s,A_t,...} \bigg[ \sum_{t'=t}^{\infin}\gamma^{t'-t}R(S_{t'}=s,A_{t'},S_{t'+1})\bigg],\\
q_{\pi}(s,a)&\doteq  \mathbb E_{\pi}\bigg[ G_t|S_t=s,A_t=a\bigg] \\
&=\mathbb E_{S_t=s,A_t=a,...}\bigg[\sum_{t'=t}^{\infin}\gamma^{t'-t}R(S_{t'}=s,A_{t'}=a,S_{t'+1}) \bigg]\end{aligned}
$$
​	**优势函数**，

$$
\begin{aligned}A_{\pi}(s,a)=\underbrace{q_{\pi}(s,a)}_{\text{On-Policy} \newline }-\underbrace{v_{\pi}(s)}_{\text{On-Policy}}\end{aligned}
$$

	## 1.3 策略迭代



# 2.蒙特卡洛与时序差分

​	**Policy-based.**基于策略的方法就是直接训练得到某个策略$\pi$。

​	**value-based.**基于值的方法不是直接学习策略，而是学习一个状态价值函数用于评估某个状态的价值从而引导智能体做出决策。在基于价值的方法中，不需要学习策略，而是由我们提前制定好智能体的行为，比如如果我们想指导状态加之后选择奖励最大的那个动作，我们可以用贪心策略。即策略是提前定义好的一个简单的函数。基于价值的方法也分为两种：

- **状态价值.**只计算状态$S_t$的价值。
- **动作价值.**计算的是状态-动作对$(S_t,A_t)$的价值。即某个状态$S_t$做出了某个$A_t$后的价值。

## 2.1 Monte Carlo

​	蒙特卡洛方法（Monte Carlo Methods）是一类基于经验采样（episode sampling）的策略评估和改进方法，它不依赖于环境模型，而是通过多次完整地与环境交互，采样出回报，进而估计状态值函数或动作值函数。

​	以状态价值函数为例，$V_\pi(s)=\mathbb E_{\pi}\bigg[ G_t|S_t=s\bigg]$，如果要计算当前状态价值需要按照预先选择的策略$\pi$生成完整的多幕轨迹，然后计算每一幕轨迹得到的$G_t^{(i)}$​，最后再进行平均，由于在一幕中某个状态$s$可能出现多次，因此又分成了首次计算（first-visit）和每次计算（every-visit）的方式，首次访问型的蒙特卡洛方法公式如下：
$$
V_\pi(s)=\frac{1}{N(s)}\sum_{i=1}^{N(s)}G_t^{(i)}
$$
​	其中 $N(s)$是状态$s$在所有幕中被第一次访问的次数。对于每一幕，只在**该状态第一次出现**时才记录它的回报$G_t$。但对于上述公式，其意味着我们需要保留所有的历史回报，得到了多幕轨迹后再执行一次更新，代价高，效率低。因此可以用增量更新的方式解决该问题，即只保留当前平均值和样本数进行更新，即当前估计值是$V_\pi^{n}(s)$，观察到第$n+1$个样本的汇报$G_{n+1}$，那么更新后的平均应该是：
$$
\begin{aligned}V_\pi^{n+1}(s)&=\frac{1}{n+1}\big(\sum_{i=1}^{n}G_i+G_{n+1}\big)\\
&=\frac{1}{n+1}(nV_\pi^{n}(s)+G_{n+1})\\
&=V_\pi^{n}(s)-\frac{1}{n+1}(G_{n+1}-V_\pi^{n}(s))\end{aligned}
$$
​	故最终状态价值函数的更新公式可以整理为：
$$
V_\pi(s)\leftarrow V_\pi(s)+\frac{1}{N(s)}\big[G_t-V_\pi(s)\big]
$$
​	有些时候我们也用一个固定的学习率$\alpha$来更新：	
$$
V_\pi(s)\leftarrow V_\pi(s)+\alpha\big[G_t-V_\pi(s)\big]
$$
​	如果$\alpha=\frac{1}{N(s)}$，那就是逐步平均（无偏）,如果固定$\alpha=0.1$，那就是更“健忘”的估计方法，可以应对非平稳环境，但有偏差。

## 2.2 Temporal Differ

## 2.3 Q-Learning

​	**Q-Learning** 是强化学习中最经典、最常见的算法之一，它是一种 **基于值函数的、离策略（off-policy）时序差分（TD）学习方法**，用于学习最优策略，即如何在不完全知道环境模型的情况下学习最优的动作选择方式。

​	Q-Learning 直接学习一个**动作价值函数（Q函数）**：
$$
xxxx
$$
​	最终目标是学习出**最优动作价值函数**$Q^*(s,a)$，从而得到最优策略：
$$
\pi^*(s,a)=\arg\max_{a}Q^*(s,a)
$$

### 2.3.1 更新

​	Q-Learning的更新公式：假设没经历一个时间步$t$，观察到当前的状态$S_t$，$A_t$和奖励$R_{t+1}$转移到新的状态$S_{t+1}$，那么更新$Q$值公式如下：
$$
Q(S_t,A_t)\leftarrow Q(S_t,A_t)-\alpha \big[R_{t+1}+\gamma\max_{a'}Q(S_{t+1},a')-Q(S_t,A_t)\big]
$$
​	$\max_{a′}Q(S_{t+1},a′)$​：**下一个状态中，估计的最大Q值**，与是否实际选择这个动作无关 → 这个公式体现了 TD 学习的本质：**用目标值对当前估计进行校正**。

​	为什么Q-Learning是离线，原因是Q-Learning训练时生成一幕数据的策略是$\epsilon-\text{greedy}$，也就是说有一定几率进行探索，但是在推理时实际上时完全采用贪心策略，每个状态下选择动作价值最大的$a$​执行动作。

### 2.3.2 具体细节

​	以$\text{maze}$游戏为例，老鼠需要吃到格子里的奶酪，如下图x：

![Maze-2](E:\Study\gitpro\KnowLedge-bak\RL系列\assets\Maze-2.jpg)

​	一共有$6$个格子，故有$6$个状态空间。最多有$4$个动作，因此初始化$\text{Q-table}$为$6\times 4$的表格，格子里面的元素初始为$0$，代表当前状态执行动作后的预期收益。$\text{Q-Learning}$具体算法如下：

**Q-Learning 算法（SarsaMax）**

---

**输入：**策略 \( $\pi$ \)，正整数 \( $\text{num\_episodes}$ \)：训练轮数，学习率 $\alpha \in (0, 1]$，探索策略（如 GLIE）对应的 $\{ \epsilon_i \}$ 

**输出：**动作价值函数 $Q(s, a)$，近似最优 $q_{\pi}$（如果训练次数足够多）

初始化：对所有状态-动作对初始化$Q(s,a)=0$，对于终止状态$s_{\text{terminal}}$，设 $Q(s_{\text{terminal}}, \cdot) = 0$

主循环：对每个 Episode（共 $\text{num\_episodes}$ 次）
>设 $\epsilon \leftarrow \epsilon_i $
>观察初始状态 $S_0$
>设$t \leftarrow 0$
>**重复执行以下步骤直到 Episode 终止：**
>
>>使用当前策略（例如$\epsilon$​​-greedy）根据$Q$​​选择动作$A_t$​​
>>执行动作$A_t$​​，观察奖励$R_{t+1}$​​和下一个状态$S_{t+1}$​​
>>$\begin{aligned}Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t) \right]\end{aligned}$​​​
>>$t \leftarrow t+1$​

结束
**返回$Q$ **

---

​	也就是说先要设定生成多少幕数据，针对每一幕，从$0$时刻开始，根据给定的策略$\pi$做出动作，每个时刻$t$都根据公式进行$\text{Q-Table}$内对应元素的更新，指导遇到终止时刻或者达到最大的幕数则当前一幕结束，技术下一幕地生成。$Q$学习实际上在时刻$t$并没有真正地做出动作，而是选择了$\arg\max_a'Q(S_t,a')$达到最大的那个动作，因此它**学习的是一条“理想化”的轨迹**，而不是自己真实经历的轨迹。它的学习目标更明确，但是因为行为与目标不一致会导致某些环境中策略收敛可能不稳定。需要足够多地探索，否则可能过早陷入次优策略。[代码示例](https://huggingface.co/learn/deep-rl-course/unit2/hands-on)如下：

```python
def train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable):
    for episode in tqdm(range(n_training_episodes)):
        # Reduce epsilon (because we need less and less exploration)
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        # Reset the environment
        state, info = env.reset()
        step = 0
        terminated = False
        truncated = False

        # repeat
        for step in range(max_steps):
            # Choose the action At using epsilon greedy policy
            action = epsilon_greedy_policy(Qtable, state, epsilon)

            # Take action At and observe Rt+1 and St+1
            # Take the action (a) and observe the outcome state(s') and reward (r)
            new_state, reward, terminated, truncated, info = env.step(action)

            # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            Qtable[state][action] = Qtable[state][action] + learning_rate * (
                reward + gamma * np.max(Qtable[new_state]) - Qtable[state][action]
            )

            # If terminated or truncated finish the episode
            if terminated or truncated:
                break
            # Our next state is the new state
            state = new_state
    return Qtable
```

## 2.4 SARSA

​	SARSA是用于估计动作价值函数$Q(s,a)$的时序差分控制算法，属于$\text{On-Policy}$，其名字来源于更新公式中用到的五元组：
$$
S_t,A_t,R_{t+1},S_{t+1},A_{t+1}
$$

**SARSA 算法**

---

**输入：**策略 \( $\pi$ \)，正整数 \( $\text{num\_episodes}$ \)：训练轮数，学习率 $\alpha \in (0, 1]$，探索策略（如 GLIE）对应的 $\{ \epsilon_i \}$ 

**输出：**动作价值函数 $Q(s, a)$，近似最优 $q_{\pi}$（如果训练次数足够多）

初始化：对所有状态-动作对初始化$Q(s,a)=0$，对于终止状态$s_{\text{terminal}}$，设 $Q(s_{\text{terminal}}, \cdot) = 0$

主循环：对每个 Episode（共 $\text{num\_episodes}$ 次）
>设 $\epsilon \leftarrow \epsilon_i $
>观察初始状态 $S_0$
>设$t \leftarrow 0$
>**重复执行以下步骤直到 Episode 终止：**
>
>>使用当前策略（例如$\epsilon$​​-greedy）根据$Q$​​选择动作$A_t$​​
>>执行动作$A_t$​​，观察奖励$R_{t+1}$​​和下一个状态$S_{t+1}$​​
>>$\begin{aligned}Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \right]\end{aligned}$​​​
>>$t \leftarrow t+1$​

结束
**返回$Q$ **

---

​	实际上也就是$Q$值的更新方式不同，由$\arg\max_{a'}Q(S_t,a')$变成了$Q(S_{t+1},A_{t+1})$​，即基于智能体真正执行的动作，因此在学习的是智能体实际走的路，在探索时会记录探索带来的结果，因此在高风险环境中更加安全。



## 2.5 DQN

​	**DQN 是 Q-Learning 的“深度版”** —— 用神经网络代替 Q 表格，解决高维状态空间问题（如图像、复杂环境等），实现“端到端”控制学习。传统 Q-learning 用表格存储 Q 值，但存在如下问题：

- 状态空间太大（如图像输入），表格根本存不下
- 状态是连续变量时，没法枚举所有状态
- 所以必须用函数逼近（Function Approximation）

​	**DQN**的核心做法是用深度神经网络逼近$Q$函数：
$$
Q_{\theta}(s,a)\approx Q^*(s,a)
$$
​	DQN 不只是把$Q$表换成神经网络，更引入了以下两个**关键机制**，用来解决 Q-learning 函数逼近时的不稳定问题。

- 经验回放(Experience Replay)

  - 训练时不用最新的数据，而是把交互数据放到一个buffer中，每次训练时从中随机采样一个小批量。

- 目标网络(Target Network)

  - Q-Learning的更新使用的是当前网络的最大$Q$值，会导致高估和不稳定，$\mathrm{DQN}$使用一个延迟更新的副本网络$Q'$计算目标：
    $$
    y=R_{t+1}+\gamma\max_{a'}Q'_{\theta^{old}}(S_{t+1},a')
    $$
    每间隔固定步数将主网络参数复制给目标网络$\theta^{old}$​。

### 2.5.1 算法流程



# 3.策略梯度算法

​	我们现在来看一下什么是策略梯度算法，以及其背后的动机与直觉。我们希望我们的策略能够使得回报的期望最大，动作的轨迹是在策略$\pi_{\theta}$的控制下产生的，因此我们的目标函数可以表达如下：
$$
\begin{aligned}J(\pi_{\theta})&=\mathbb E_{\tau \sim(\pi_{\theta},E)}\bigg[G(\tau)\bigg] \\
&=\sum_{\tau \sim (\pi_{\theta},E)} P(\tau|\theta)G(\tau)\end{aligned} \tag{2.1}
$$
​	其中$E$是环境，这个目标函数衡量了在指定环境下从我们的策略中采样的轨迹的理论收益。如果我们想找到最大化这个目标函数的参数$\theta$，我们可以通过梯度上升的方式不断迭代更新$\theta$。更新过程表示如下：
$$
\theta_{t+1}=\theta_t+\alpha \nabla_{\theta} J(\pi_{\theta})|_{\theta_t} \tag{2.2}
$$
​	其中$\nabla_{\theta} J(\pi_{\theta})|_{\theta_t}=\nabla_{\theta_t} J(\pi_{\theta_t})$，也就是所谓的**策略梯度**。现在的问题在于，我们如何计算$\nabla_{\theta} J(\pi_{\theta})|_{\theta_t}$​？利用策略梯度则需要一个明确的可计算的表达式。接下来我们进一步探索如何找到这个明确的表达式并加以运用。

​	**1.轨迹的概率**，给定策略$\pi_{\theta}$下轨迹$\tau=(s_0,s_1,...,s_{T+1})$出现的概率如下：
$$
P(\tau|{\theta})=\rho(S_0)\prod_{t=0}^{T}P(S_{t+1}|S_t,A_t)\pi_{\theta}(A_t|S_t) \tag{2.3}
$$
​	**2.Log-Derivative Trick**，利用$\log$求导的技巧$\part_x \log(f(x))=\frac{\partial_x f(x)}{f(x)}$，对轨迹概率关于$\theta$求导，我们有：
$$
\nabla_{\theta}P(\tau|{\theta})=\nabla_{\theta}\log {P(\tau|\theta)}\cdot P(\tau|\theta) \tag{2.4}
$$
​	**3.轨迹的对数概率**，$P(\tau|\theta)$​的对数概率如下：
$$
\log P(\tau|\theta)=\underbrace{\log(\rho(S_0))}_{与\theta 无关}+\sum_{t=0}^{T}\underbrace{\log {P(S_{t+1}|S_,A_t)}}_{与\theta 无关}+\log(\pi_{\theta}(A_t|S_t)) \tag{2.5}
$$
​	**4.轨迹对数概率的梯度**，由于部分项与$\theta$无关，所以$\nabla_{\theta}\log P(\pi|\theta)$表达如下：
$$
\begin{aligned}\nabla_{\theta}\log P(\tau|\theta)&=\nabla_{\theta}\sum_{t=0}^{T}\log (\pi_{\theta}(A_t|S_t)) \\&=\sum_{t=0}^{T}\nabla_{\theta}\log (\pi_{\theta}(A_t|S_t))\end{aligned} \tag{2.6}
$$
​	将上述式子结合，求出$\nabla_{\theta}J(\pi_{\theta})$​则有：
$$
\begin{aligned} \nabla_{\theta}J(\pi_{\theta})&=\nabla_{\theta}\sum_{\tau \sim (\pi_{\theta},E)}P(\tau|\theta)G(\tau) \\
&=\sum_{\tau \sim (\pi_{\theta},E)}\nabla_{\theta}P(\tau|\theta)G(\tau) \\
&=\sum_{\tau \sim (\pi_{\theta},E)}\nabla_{\theta}\log P(\tau|\theta) P(\tau|\theta)R(\tau) \\
&=\sum_{\tau \sim (\pi_{\theta},E)}\sum_{t=0}^{T}\nabla_{\theta}\log (\pi_{\theta}(A_t|S_t)) P(\tau|\theta)G(\tau) \\
&=\mathbb E_{\tau \sim (\pi_{\theta},E)} [\sum_{t=0}^{T}\nabla_{\theta}\log (\pi_{\theta}(A_t|S_t))G(\tau)] \end{aligned} \tag{2.7}
$$
​	也就是说，我们可以根据采样到的轨迹来计算策略梯度，即如果收集到了一系列由$\pi_{\theta}$产生的轨迹$D=\set{\tau_i}_{i=1}^{N}$，我们可以利用如下公式估计策略梯度：
$$
\hat {\nabla_{\theta}}J(\pi_{\theta})=\frac{1}{|D|}\sum_{i=1}^{N}\sum_{t=0}^{T}\nabla_{\theta}\log (\pi_{\theta}(A_t|S_t))G(\tau) \tag{2.8}
$$
​	如上便是策略梯度最简单的形式。如果我们能够在环境中运行策略来收集轨迹数据集，我们可以计算策略梯度并采取更新步骤。

​	**Don't Let the Past Distract You**.策略梯度的公式告诉我们对于一个轨迹$\tau=(s_0,s_1,...,s_T)$，每一个时刻的动作执行后对应的策略梯度都会乘以一个因子，即回报$R(\tau)$，但这并没有什么意义，因为是否强化智能体当前的决策动作只因和当前动作执行后产生的影响有关，如果当前动作执行后的收益低则不应当强化当前动作，反之亦然，而不能受先前因素的影响。因此，对于回报$R(\tau)$我们可以做出适当改进，不再考虑当前时刻$t'$之前的收益，则策略梯度可以重写为：
$$
\begin{aligned} \nabla_{\theta}J(\pi_{\theta})
&=\mathbb E_{\tau \sim (\pi_{\theta},E)} [\sum_{t=0}^{T}(\nabla_{\theta}\log (\pi_{\theta}(A_t|S_t) \sum_{t=t'}^{T}R(S_{t'},A_{t'},S_{t'+1}))] \end{aligned} \tag{3.10}
$$
​	在这个形式下，动作只基于在采取行动后获得的奖励而得到强化。我们将这种形式称为 **“Reward-to-go”**，因为回报为在轨迹的一个点之后奖励的总和。

​	策略梯度算法有一些变种形式，这些变种都与我们先前所学习的内容有所关联，接下来我们学习一些变种算法从而对策略梯度算法有更深入的理解。

​	**Baseline in Policy Gradients.** 使用策略梯度面临着一个问题——准确地估算出梯度需要大量的样本轨迹，使用 EGLP 引理，我们可以证明**Reward -to-go**——尽管没有改变政策梯度的期望值——减少了我们估计的方差，因此减少了估计策略梯度所需的轨迹总数。而EGLP的直接结果是对于任何直接依赖于状态的函数$b$，我们有：
$$
\begin{aligned} \mathbb E_{A_t \sim \pi_{\theta}} [\nabla_{\theta}\log(\pi_{\theta}(A_t|S_t)b(s_t)]=0\end{aligned} \tag{3.11}
$$
​	这使得我们可以在不改变期望值的情况下在策略梯度表达式上加上或减去任意项：
$$
\begin{aligned} \nabla_{\theta}J(\pi_{\theta})
&=\mathbb E_{\tau \sim (\pi_{\theta},E)} [\sum_{t=0}^{T}\bigg(\nabla_{\theta}\log (\pi_{\theta}(A_t|S_t) \big(\sum_{t=t'}^{T}R(S_{t'},A_{t'},S_{t'+1})-\underbrace{b(s_t)}_{\text{Baseline}}\big)\bigg)] \end{aligned}
\tag{3.12}
$$
​	在这个表达式中，任何函数$b$都称作$\text{baseline}$。最常见的基线选择是状态价值函数$V^{\pi}(s_t)$。回想一下，这是一个智能体从状态开始，然后在其剩余生命周期内按照策略行事时获得的平均回报。从**经验上**讲，这种选择具有减少策略梯度样本估计方差的效果，使得策略梯度学习更快更稳定。从直观上将，假设在某个状态$S_t$下，智能体采取了一个动作 $A_t$，并得到了一个回报。如果这个回报远高于该状态的预期回报（即 $V^{\pi}(s_t)$），我们认为这个动作“更好”；如果低于预期回报，我们认为该动作“不好”，通过减去$V^{\pi}(s_t)$​，我们将焦点集中在“超出预期的部分”（即优势）上。这就像在与基准相比时，我们只关注某个动作相对于平均策略的改进或削弱。

> [!NOTE]
>
> 值得注意的是，$V^{\pi}(s_t)$并不能准确的计算，所以需要近似计算，通常我们用一个神经网络$V_{\phi}^{\pi}(s_t)$估计价值函数，它与策略网络同时更新，而$V_{\phi}^{\theta}$的优化目标通常是最小均方误差（包括$VPG,TRPO,PPO$等），即：
> $$
> \phi_k= \arg\min_{\phi} \mathbb E_{S_t,\hat{R_t}\sim \pi_k}\big[ \big( V_{\phi}(S_t)-\hat{R_t}\big)^2 \big] \tag{3.13}
> $$

​	我们可以以一种更一般地形式写出策略梯度：
$$
\begin{aligned} \nabla_{\theta}J(\pi_{\theta})
&=\mathbb E_{\tau \sim (\pi_{\theta},E)} \bigg[ \sum_{t=0}^{T}\bigg(\nabla_{\theta}\log (\pi_{\theta}(A_t|S_t) \Psi(t)\bigg) \bigg] \end{aligned} \tag{3.14}
$$
​	$\Psi(t)=R(\tau)$时$\nabla_{\theta}J(\pi_{\theta})$是基础，$\Psi(t)=\sum_{t=t'}^{T}R(S_{t'},A_{t'},S_{t'+1})$时是**Reward-to-go**，$\Psi(t)=\sum_{t=t'}^{T}R(S_{t'},A_{t'},S_{t'+1})-b(S_t)$是**Reward-to-go with baseline**。此外，$\Psi(t)$的选择还可以是动作价值函数$Q^{\pi}(S_t,A_t)$，优势函数$A^{\pi}(s_t,a_t)=Q^{\pi}(S_t,A_t)-V^{\pi}(S_t)$​，利用优势函数的策略梯度的公式化极为常见，并且不同算法使用的优势函数有许多不同的估计方法。

​	上文所述$V^{\pi}(S_t)$并不能准确的计算，需要引入基于价值的方法，基于价值的方法就是$\text{DQN}$，$\text{DQN}$也有两种不同的价值估计方式，分别是状态价值和动作价值。如果是用神经网络估计状态价值，那么就是估计$V^{\pi}(s_t)$，动作价值则是估计$$Q^{\pi}(S_t,A_t)$$。

## 3.1 Actor-Critic

​	随机变量$G$的期望值就是当前的动作价值，即：
$$
\mathbb E[G_t]=\sum_{t'=t}^{T}\gamma ^{t'-t}R(S_{t'},A_{t'},S_{t'+1})=Q(S_t,A_t)
$$
​	若基线选取$V_{\pi}(s_t)$，那么有：
$$
\begin{aligned} \nabla_{\theta}J(\pi_{\theta})
&=\mathbb E_{\tau \sim (\pi_{\theta},E)} [\sum_{t=0}^{T}\bigg(\nabla_{\theta}\log (\pi_{\theta}(A_t|S_t) \big(Q_\pi(S_t,A_t)-\underbrace{V_{\pi}(S_t)}_{\text{Baseline}}\big)\bigg)] \end{aligned}
\tag{3.12}
$$
​	该算法称为优势演员-评论员算法。表明上这个公式需要我们训练$Q$网络和$V$​网络，但是实际上动作价值可以由状态价值直接表示，如公式(xxx)有：
$$
Q_\pi(s_t,a_t)=\sum_{r_t,s_{t+1}}p(r_t,s_{t+1}|s_t,a_t)(r_t+\gamma v_{\pi}(s_{t+1}))=\mathbb E[r_t+\gamma V^\pi(s_{t+1})]
$$
​	可以看到有一个求期望的操作，如果把求期望的符号去除，用单次得到的结果近似$Q$值，那么此时就有：
$$
Q^\pi(s_t,a_t)-V^\pi(s_t)=r_t+\gamma V^\pi(s_{t+1})-V^\pi(s_t)
$$

> [!NOTE]
>
> 为什么把估计期望这一步去掉？原论文进行的尝试，发现效果还不错，理论上没有严格证明。

​	所以最后的Actor-Critic算法的损失如下：
$$
\begin{aligned} \nabla_{\theta}J(\pi_{\theta})
&=\mathbb E_{\tau \sim (\pi_{\theta},E)} [\sum_{t=0}^{T}\bigg(\nabla_{\theta}\log (\pi_{\theta}(A_t|S_t) \big(R_t+\gamma V^{\pi}(S_{t+1})-\underbrace{V^{\pi}(S_t)}_{\text{Baseline}}\big)\bigg)] \end{aligned}
\tag{3.12}
$$
​	演员网络和评论员网络的输入都是状态$s$，所以它们前面几个层（layer）是可以共享的。如在RLHF中，Critic Model就是由奖励模型在最后一层换成一个可训练的线性层得到的。此外，为了使模型能够做到探索，通知会对策略的输出分布设置一个约束，使得熵不要太小凹（熵太小意味着分布比较确定，可能会集中在某几个动作上），这样智能体才会尝试多种不同动作，充分探索环境。

## 3.2TRPO

​	```为保证上下文符号一致，笔者在本章节推导上的符号并未遵循原论文， 进行了一定改动。```

​	一个策略$\tilde \pi$关于另一个策略$\pi$的预期收益在累计时间步上的优势为：
$$
J(\tilde \pi)-J(\pi)=\mathbb E_{\tau\sim \tilde \pi}\bigg[ \sum_{t=0}^{\infin}\gamma^tA_{\pi}(S_t,A_t)\bigg]\tag{3.15}
$$
​	在$TRPO$的原论文中，提供了该公式的反向证明如下（笔者略微进行了调整）：
$$
\begin{aligned} \mathbb E_{\tau \sim \tilde \pi}&\bigg[ \sum_{t=0}^{\infin}\gamma^tA_{\pi}(S_t,A_t)\bigg] \\
&=\mathbb E_{\tau \sim \tilde \pi}\bigg[ \sum_{t=0}^{\infin}\gamma^t(q_{\pi}(S_t,A_t)-v_{\pi}(S_t))\bigg] \\
&=\mathbb E_{\tau \sim \tilde \pi}\bigg[ \sum_{t=0}^{\infin}\gamma^t(R(S_t,A_t,S_{t+1})+\gamma v_{\pi}(S_{t+1})-v_{\pi}(S_t))\bigg]\\
&=\mathbb E_{\tau \sim \tilde \pi}\bigg[ \sum_{t=0}^{\infin}\gamma^tR(S_t,A_t,S_{t+1})+\gamma^{t+1}v_{\pi}(S_{t+1})-\gamma^t v_{\pi}(S_t)\bigg]\\
&=\mathbb E_{\tau \sim \tilde \pi}\bigg[ \sum_{t=0}^{\infin}\gamma^tR(S_t,A_t,S_{t+1})\bigg]+\mathbb E_{\tau \sim \tilde \pi}\bigg[\sum_{t=0}^{\infin}\gamma^{t+1}v_{\pi}(S_{t+1})-\gamma^t v_{\pi}(S_t)\bigg]\\
&=J(\tilde \pi)+\mathbb E_{\tau \sim \tilde \pi}\bigg[\sum_{t=1}^{\infin}\gamma^{t}v_{\pi}(S_{t})-\sum_{t=0}^{\infin}\gamma^t v_{\pi}(S_t)\bigg]\\
&=J(\tilde \pi)-\mathbb E_{\tau \sim \tilde \pi}\bigg[v_{\pi}(S_0)\bigg]\\
&=J(\tilde \pi)-J(\pi)\end{aligned}\tag{3.16}
$$

​	反过来推导显得思路不那么自然，而直接记住这个结论显得略微生硬，先抛开繁琐的证明，我们先通过图像来看看一个有限马尔可夫决策过程如何计算其预期收益，再理解什么是一个策略$\tilde \pi$关于另一个策略$\pi$在累计时间步上的收益：

![image-20250228172223808](assets\image-20250228172223808.png)

​	我们知道，预期收益一定是一个和的形式，最显而易见的思路是在时间步上求和（上图右部分），即计算每一个时间步$t$的预期收益，按照时间步不断累加得到总的预期收益。此外，我们还可以从状态空间的视角来解决这个问题，即上图的中间部分，我们先遍历状态$s\in \mathcal S$，当状态轨迹$s_1\rightarrow s_1\rightarrow s_1 \cdots\rightarrow s_1$固定时，我们会得到一条时间轨迹（虚线部分）。









如果能保证$J(\tilde \pi)-J(\pi)$大于$0$，则能说明更新后的策略一直在进步，而优势函数这一项又可以改写成：
$$
\begin{aligned}\mathbb E_{\tau \sim \tilde \pi}&\bigg[ \sum_{t=0}^{\infin}\gamma^tA_{\pi}(s_t,a_t)\bigg] \\
&=\sum_{t=0}^{\infin}\sum_{s}p(s_t=s|\tilde \pi)\sum_a \tilde \pi(a_t=a|s)\gamma^tA_{\pi}(s,a)\\
&=\sum_{t=0}^{\infin}\sum_{s}\gamma^tp(s_t=s|\tilde \pi)\sum_a \tilde \pi(a_t=a|s)A_{\pi}(s,a)\\
&=\sum_{s}\sum_{t=0}^{\infin}\gamma^tp(s_t=s|\tilde \pi)\sum_a \tilde \pi(a_t=a|s)A_{\pi}(s,a)\\
&=\sum_{s}\rho_{\tilde \pi}(s)\sum_a \tilde \pi(a_t=a|s)A_{\pi}(s,a)\end{aligned}\tag{3.17}
$$
​	其中，$\rho_{\tilde \pi}(s)=p(s_0=s|\tilde \pi)+\gamma p(s_1=s|\tilde \pi)+,...，$$\tilde \pi$是之前的策略$\pi$更新后的新策略，上述式子中涉及到$p(s_t=s|\tilde \pi)$与$\tilde \pi(a_t=a|s)$，即我们要按照新的策略与环境交互才能得到轨迹，先确定新的策略$\tilde \pi$并得到一定量的样本才能求解，并计算是否满足$\mathbb E_{\tau \sim \tilde \pi}\bigg[ \sum_{t=0}^{\infin}\gamma^tA_{\pi}(s_t,a_t)\bigg]\gt0$。$TRPO$利用函数$\mathcal L_{\pi}(\tilde \pi)$代替原始目标函数：
$$
\mathcal L_{\pi}(\tilde{\pi})=J(\pi)+\sum_{s} \rho_{\pi}(s) \sum_{a} \tilde{\pi}(a \mid s) A_{\pi}(s, a) .\tag{3.18}
$$
​	只要策略更新的幅度不大，就可以用$\mathcal L_{\pi}(\tilde \pi)$近似原本的$J(\tilde \pi)$，所以那我们**怎么来保证其更新幅度不要太大呢**？为了解决这个求解信任区域的问题，文中引入了Kakade&Langford（2002）的结论——Conservative policy iteration：

$$
\begin{aligned}\pi_{\text {new }}(a \mid s)&=(1-\alpha) \pi_{\text {old }}(a \mid s)+\alpha \pi^{\prime}(a \mid s) \\
\eta\left(\pi_{\text {new }}\right) & \geq L_{\pi_{\text {old }}}\left(\pi_{\text {new }}\right)-\frac{2 \epsilon \gamma}{(1-\gamma)^{2}} \alpha^{2} \\
& \text { where } \epsilon=\max _{s}\left|\mathbb{E}_{a \sim \pi^{\prime}(a \mid s)}\left[A_{\pi}(s, a)\right]\right|\end{aligned}\tag{3.19}
$$
​	有了这个下界表达式，我们可以利用**minorization-maximization**算法通过$\mathcal L_{\pi_{old}}(\pi_{new})$迭代$J(\pi_{new})$​。该算法具体细节不在本文涉及范围内，值得注意的是，该原始结论只适合混合策略，但实际应用中的混合策略很少使用，因此作者将该结论拓展到了一般随机策略[x]。最终的优化目标变成：

$$
\underset{\pi_{\theta}}{\operatorname{maximize}}\left[\mathcal L_{\pi_{\theta_{\text {old }}}}(\pi_{\theta})-C D_{\mathrm{KL}}^{\max }\left(\pi_{\theta_{\text {old }}}, \pi_{\theta}\right)\right] .\tag{3.20}
$$
​	其中：
$$
\mathcal L_{\pi_{old}}({\pi}_{\theta})=J(\pi)+\sum_{s} \rho_{\pi_{\theta_{old}}}(s) \sum_{a} {\pi_{\theta}}(a \mid s) A_{\pi_{\theta_{old}}}(s, a) .
$$
​	由于$\pi_{\theta}(a|s)$​与新策略有关，无法对其直接采样，因此我们通过重要性采样的方式进行采样，我们将式子右边项进行变体：
$$
\begin{aligned}\sum_{s} &\rho_{\pi_{\theta_{old}}}(s) \sum_{a} {\pi_{\theta}}(a \mid s) A_{\pi_{\theta_{old}}}(s, a)\\
&=\sum_{s}\sum_{t} \gamma^t p(s_t=s|\pi_{\theta_{old}})\sum_{a} {\pi_{\theta}}(a \mid s) A_{\pi_{\theta_{old}}}(s, a)\\
&=\sum_{t}\gamma^t \sum_{s} p(s_t=s|\pi_{\theta_{old}})\sum_{a} {\pi_{\theta}}(a \mid s) A_{\pi_{\theta_{old}}}(s, a)\\
&=\sum_{t}\gamma^t \mathbb E_{s\sim\rho_{old}} \bigg[ \sum_a{\pi_{\theta}}(a \mid s) A_{\pi_{\theta_{old}}}(s, a)\bigg]\\
&=\frac{1}{1-\gamma} \mathbb E_{s\sim\rho_{old}} \bigg[ \sum_a{\pi_{\theta}}(a \mid s) A_{\pi_{\theta_{old}}}(s, a)\bigg]\end{aligned}\tag{3.21}
$$
​	$\sum_a{\pi_{\theta}}(a \mid s) A_{\pi_{\theta_{old}}}(s, a)$​可以通过重要性采样的方式重新表述成：
$$
\begin{aligned}\sum_a{\pi_{\theta}}(a \mid s) A_{\pi_{\theta_{old}}}(s, a)&=\mathbb E_{a \sim q}\bigg[ \frac{{\pi_{\theta}}(a \mid s)}{q_(a \mid s)} A_{\pi_{\theta_{old}}}(s, a)\bigg]\\
&\mapsto\mathbb E_{a \sim \pi_{\theta_{old}}}\bigg[ \frac{{\pi_{\theta}}(a \mid s)}{\pi_{\theta_{old}}(a \mid s)} A_{\pi_{\theta_{old}}}(s, a)\bigg]\end{aligned}
$$
​	故最终的优化目标为：
$$
\begin{aligned} \arg\max_{\theta}&\mathbb E_{s\sim \rho_{old},a\sim \pi_{old}}\bigg[ \frac{{\pi_{\theta}}(a \mid s)}{\pi_{\theta_{old}}(a \mid s)} A_{\pi_{\theta_{old}}}(s, a)\bigg] \\
&{\operatorname {subject to}}{\text{ }} \mathbb E_{s\sim \rho_{old}}\bigg[ D_{KL}(\pi_{\theta_{old}}(\cdot|s)||\pi_{\theta}(\cdot|s))\bigg] \leq \delta
\end{aligned}\tag{3.22}
$$

## 3.2 PPO 

​	我们现在将深入理解为语言模型对齐奠定基础的算法 —— $PPO$(近端策略优化算法)，$PPO$是一种基于策略梯度的强化学习算法，$PPO$的核心思想是通过在每次更新时保持策略的“平稳性”或“稳定性”，避免过度优化，从而减少策略更新过程中的波动性，$PPO$算法的优化目标如下（单个样例）：
$$
\left.J(\theta)=\frac{1}{T} \sum_{t=1}^{T} \min \left(\frac{\pi_{\theta}\left(a_{t} \mid s_t\right)}{\pi_{\theta_{o l d}}\left(a_{t} \mid s_t\right)} A_{t}, \operatorname{clip}\left(\frac{\pi_{\theta}\left(a_{t} \mid s_t\right)}{\pi_{\theta_{o l d}}\left(a_{t} \mid s_t\right)}, 1-\varepsilon, 1+\varepsilon\right) A_{t}\right)\right)
$$
​	注意到$\min$函数会使优化目标有不同取值的选择，我们来解释不同情况下的取值情况，首先定义策略比率如下：
$$
R(\theta)=\frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}
$$
​	当优势是正数且策略比率超过$1+\varepsilon$时，意味着新的策略更可能采取$a_t$行动，$\operatorname{clip(\cdot)}$通过裁剪比率防止策略更新过大，并限制$R(\theta)$的变化范围，此时优化目标：
$$
J(\theta)=\min(R(\theta),1+\varepsilon)A_t=(1+\epsilon)A
$$
​	意味着优化目标会增加策略比率，使得动作更可能发生，当优势是正数而策略比率小于$1-\varepsilon$​时，优化目标变成：
$$
J(\theta)=\min(R(\theta),1-\varepsilon)A_t=R(\theta)A_t
$$
​	意味着优化目标会减小策略比率，使得动作没那么可能发生。同理，如果策略比率本身介于$(1-\varepsilon,1+\varepsilon)$之间，则有：
$$
J(\theta)=\min(R(\theta),R(\theta))=R(\theta)
$$
​	没有任何影响，当优势是负数时且策略比率$R(\theta)\lt 1-\varepsilon$时，有：
$$
J(\theta)=\min(R(\theta)A_t,(1-\varepsilon)A_t)=(1-\varepsilon)A_t
$$
​	其他情况同理，在信任区域内的优化目标与策略梯度是一致的。

接下来我们将结合部分的代码深入理解$PPO$算法如何在$RLHF$中大展身手，如何完成大语言模型对齐的任务。

### 3.2.1  DeepSpeed-Chat 实现

​	在$\mathrm{training/step3\_rlhf\_finetuning/main.py}$下的$for$循环内有两个比较重要的函数，第一个是第$537$行的$\mathrm{generate\_experience}$，另一个是第$553$行的$\mathrm{train\_rlhf}$。分别用于生成轨迹以及计算损失函数。

![image-20250217155740680](assets\image-20250217155740680.png)

​	**生成轨迹.**先看第一个函数$\text{generate\_experience}$，返回的字典中包括了$actor$和$reference$的对数几率，并且还有$actor$生成的序列的奖励以及每一个$token$对应的价值。价值计算由$\text{critic\_model.forward\_value()}$产生，对应实现在dschat/utils/model/reward_model.py。函数中实现的功能如下图所示：

![image-20250219104714800](assets\image-20250219104714800.png)

​	该函数中又涉及到了三个重要的函数，由图中的橙色粗体表示，分别是$\text{\_generate\_sequence}$，作用是$actor$根据给定的$prompt$生成完整的序列，以及$\text{forward\_value}$，$\text{Reward}$和$\text{Critic}$分别生成奖励和$token$序列的价值，最后是根据给定的维度和索引收集指定的对数概率，由函数$\text{gather\_loh\_probs}$完成。

![image-20250217135919357](assets\image-20250217135919357.png)

​       在$\text{forward\_value}$函数中，目的是为了拿到模型输出的奖励，因此需要找到整个序列中只属于$answer$的这部分$token$，第$166$行得到的$c\_inds$是模型输出的回答部分结束的位置索引，第$168-169$行是拿到最后一个位置前的输出的$value$，并返回$value$序列以及最后一个位置获得的奖励$\text{chosen\_end\_scores}$。

### 3.2.2 计算PPO 损失	

​	在拿到了生成的轨迹和奖励与价值后，继续往下看到$for$循环内的第$564$行代码，在$\text{train\_rlhf}$中完成了损失函数计算和反向传播，该函数输入变量是$\text{generate\_experience}$函数的返回结果，即。

![image-20250217165451472](assets\image-20250217165451472.png)

​	右边子图是该方法的具体实现，其中又涉及到了四个重要的函数，第一个函数是计算$actor$在策略$\pi_{\theta}$下获得的奖励，第二个函数是计算策略$\pi_{\theta}$下的优势，第三个函数则是计算$PPO$损失，第四个函数则是计算$critic$损失，四个函数均定义在$\text{DeepSpeedPPOTrainer}$的类下，整体关系如下图所示：

![image-20250220114454497](assets\image-20250220114454497.png)

​	接下来看到$\text{compute\_rewards}$和$\text{get\_advantages\_and\_returns}$两个函数，分别完成了奖励计算以及优势与回报计算。$rewards$其实计算一个$[Batch,Seq]$的二维张量，$Seq$这个维度每一个元素需要计算$actor$和$reference$输出的$KL$散度即$\mathbb E_{A_t\sim\pi_{\theta}}[\log \frac{\pi_{\theta}(A_t|S_t)}{\pi_{ref}(A_t|S_t)}]$，且最后一个位置还要加上$seq$序列最终的奖励分数。需要注意的是，在$\mathrm{ppo\_epoches}$这个循环中，一开始$\mathrm {actor\_log\_probs}$与$\mathrm {log\_probs}$二者是相同的，随着循环的进行，$actor$不断更新后二者不再相同。

![image-20250219171740261](assets\image-20250219171740261.png)

​	在计算完带有惩罚项的奖励以后，我们需要计算优势函数$A_t$。在广义优势估计中，有：
$$
\begin{aligned}A_t&=\sum_{k=0}^{\infin}(\gamma \lambda)^k\delta_{t+k}\\
&=\delta_t+\underbrace {(\gamma \lambda)\sum_{k=0}^{\infin}(\gamma \lambda)^k\delta_{t+1+k}}_{\gamma\lambda A_{t+1}}\\
&=\delta_t+\gamma\lambda\delta_{t+1}+\underbrace{(\gamma \lambda)^2\sum_{k=0}^{\infin}(\gamma \lambda)^k\delta_{t+2+k}}_{(\gamma\lambda)^2 A_{t+2}}\\
&=\cdots{}\cdots{}\\
\delta_t&=r_t+\gamma V(S_{t+1})-V(S_t)\end{aligned}
$$
​	由上面公式可知，计算$A_t$需要知道$\delta_t$和$A_{t+1}$，计算$\delta_t$需要知道$r_t,V(S_{t+1})$和$V(S_t)$，而$r_t,V(S_t)$可以由图中的$\text{rewards}$和$\text {old\_values}$直接得出，所以问题在于如何求$A_{t+1}$，此时如果正向计算$A_t,t=0,1...,T$就存在一个问题，计算$A_0$需要知道$A_1$,计算$A_1$需要知道$A_2$，以此类推需要先把$A_t$全部都先算出来才能知道最开始的$A_0$（增加内存占用，需保存所有中间结果），因此我们通常采用倒序计算的方式以更高效地解决这个问题，即我们先算$A_T$，并基于如下公式递推计算$A_{T-1}=\delta_{T-1}+\gamma\lambda A_T$，每一步复用上一步的结果，最终计算完$A_0$。计算完优势序列$\mathbf A$后，根据$PPO$的公式，我们只需要再计算出新旧策略执行动作的比率$R_t=\frac{\pi_{\theta}(A_t|S_t)}{\pi_{\theta_{old}}(A_t|S_t)}$就能计算损失函数。而比率$R_t$就是更新后的$actor$输出的动作序列（对数概率）除以没更新前的$actor$的动作序列，整个流程示意图如下所示：

![image-20250221170756640](assets\image-20250221170756640.png)

​	计算$R_t$依赖于更新后的$actor$的动作概率和未更新的$actor$的动作概率，而$\text{actor\_log\_prob}$与$\text{log\_prob}$是对数概率，因此有$R_t=\exp \{ \log\pi_{\theta}(A_t|S_t)-\log{\pi_{\theta_{old}}}(A_t|S_t)\}$，再依据$PPO$损失函数公式得到最后的损失$\text{pg\_loss}$​。$actor$会不断改进策略以执行更好的动作，与此同时$critic$也需要不断更新，读者可以理解为一个教练不能总是以过往的眼光来评价一个不断进步的演员。$critic$的损失函数计算比较简单，采用的是平方差损失：

![image-20250221175504910](assets\image-20250221175504910.png)

​	首先通过$\mathrm{torch.clamp}$将$\mathrm{values}$​限制到一定范围，计算平方差损失后再取每一个位置上的最大值，如上便是整个RLHF-PPO算法的核心流程实现，值得注意的是，在$\mathrm{ppo\_epochs}$的循环中，$actor$与$critic$更新以后在$558$行还有一个无监督训练：

![image-20250221180936573](assets\image-20250221180936573.png)

​	进入$\text{train\_unsupervised}$方法后可以发现其就是默认的损失，即next token prediction loss，（数据集不是SFT形式的数据，是预训练形式的数据）目的是为了在强化学习过程中保持模型的通用领域知识，防止模型被带偏。

## 3.5 GRPO

​	GRPO是由DeepSeek团队在数学推理任务中提出的算法，在传统的$PPO$算法上改变了两点：（1）优势函数的改变，去掉了Critic Model。（2）KL散度。整体示意图如下[x]()：

![image-20250808101409438](assets\image-20250808101409438.png)

根据DeeseekMath这篇论文，PPO的损失函数写作：
$$
\mathcal{J}_{ppo}(\theta)=\mathbb{E}\left[q \sim P(Q), o \sim \pi_{\theta_{\text {old }}}(O \mid q)\right] \frac{1}{|o|} \sum_{t=1}^{|o|} \min \left[\frac{\pi_{\theta}\left(o_{t} \mid q, o_{<t}\right)}{\pi_{\theta_{\text {old }}}\left(o_{t} \mid q, o_{<t}\right)} A_{t}, \operatorname{clip}\left(\frac{\pi_{\theta}\left(o_{t} \mid q, o_{<t}\right)}{\pi_{\theta_{\text {old }}}\left(o_{t} \mid q, o_{<t}\right)}, 1-\varepsilon, 1+\varepsilon\right) A_{t}\right],
$$
​	GRPO的损失函数可以写作：
$$
\begin{aligned}
\mathcal{J}_{G R P O}(\theta) & =\mathbb{E}\left[q \sim P(Q),\left\{o_{i}\right\}_{i=1}^{G} \sim \pi_{\theta_{\text {old }}}(O \mid q)\right] \\
& \frac{1}{G} \sum_{i=1}^{G} \frac{1}{\left|o_{i}\right|} \sum_{t=1}^{\left|o_{i}\right|}\left\{\min \left[\frac{\pi_{\theta}\left(o_{i, t} \mid q, o_{i,<t}\right)}{\pi_{\theta_{\text {old }}}\left(o_{i, t} \mid q, o_{i,<t}\right)} \hat{A}_{i, t}, \operatorname{clip}\left(\frac{\pi_{\theta}\left(o_{i, t} \mid q, o_{i,<t}\right)}{\pi_{\theta_{\text {old }}}\left(o_{i, t} \mid q, o_{i,<t}\right)}, 1-\varepsilon, 1+\varepsilon\right) \hat{A}_{i, t}\right]-\beta \mathbb{D}_{K L}\left[\pi_{\theta}| | \pi_{\text {ref }}\right]\right\}
\end{aligned}
$$
​	也就是说对于一个给定的prompt或者说问题$q$，根据旧的策略$\pi_{\theta_{\text{old}}}$生成得到$G$个输出，即$\{o_i\}_{i=1}^G$，然后依据制定的规则式奖励为每个生成的$o_i$计算奖励，得到$\{r_i\}_{i=1}^G$，然后计算组内相对优势$\hat A_{i,t}$：
$$
r_i=\frac{r_i-\mathrm{mean}(\mathbf r)}{\mathrm{std}(\mathbf r)}
$$
​	因此，对于每一个生成的$o_i$，所有token都是共享奖励的。此外，GRPO在KL散度方面使用了无偏估计的方式，对于每一个token而言，有：
$$
\mathbb{D}_{K L}\left[\pi_{\theta}| | \pi_{\text {ref }}\right]=\frac{\pi_{\text{ref}}\left(o_{i, t} \mid q, o_{i,<t}\right)}{\pi_{\theta}\left(o_{i, t} \mid q, o_{i,<t}\right)}-\log\frac{\pi_{\text{ref}}\left(o_{i, t} \mid q, o_{i,<t}\right)}{\pi_{\theta}\left(o_{i, t} \mid q, o_{i,<t}\right)}-1
$$
​	也就是是说GRPO计算的KL是采样KL，只看实际生成的token序列和ref输出的log prob的差异，而PPO是计算期望状态分布上的全分布KL。



# 参考文献

[[X]Policy Gradients: The Foundation of RLHF](https://cameronrwolfe.substack.com/p/policy-gradients-the-foundation-of)

[[X]Proximal Policy Optimization (PPO): The Key to LLM Alignment](https://cameronrwolfe.substack.com/p/proximal-policy-optimization-ppo)

[[X](WIP) A Little Bit of Reinforcement Learning from Human Feedback](https://rlhfbook.com/book.pdf)

[[x]Policy Gradient Algorithms,Weng, Lilian,liliangweng.github.io,2018](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)

[[x]Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)

[[x]Approximately Optimal Approximate Reinforcement Learning](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/KakadeLangford-icml2002.pdf)

[[x]深度强化学习（三）：TRPO（Trust Region Policy Optimization ，信赖域策略优化）,Dreammaker](https://zhuanlan.zhihu.com/p/605886935)
