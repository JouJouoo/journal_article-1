# DP-LCRL 动态群体强化学习实验方案

## 1. 验证目标

本文的第二个核心创新点是：将动态参与 P2P 交易建模为可变主体集合上的序列决策问题，并通过“智能体掩码 + 自注意力”实现对时变活跃主体集合的有效编码与决策生成。

因此，本组实验不把渐进扩容、随机规模训练或最大规模训练作为核心创新点，而是集中验证以下三个问题：

1. 序列建模是否优于普通固定拼接或 MLP 策略。
2. 智能体掩码是否能有效排除非活跃主体干扰。
3. 在节点动态加入和退出更频繁时，方法是否保持稳定。

## 2. 实验总览

| 实验 | 验证内容 | 方法 | 主要对照 |
|---|---|---|---|
| 实验 1：序列建模有效性 | Transformer 自注意力是否优于普通固定拼接/MLP 策略 | 比较序列化自注意力策略与非序列建模策略 | DP-LCRL vs MLP/MAPPO baseline |
| 实验 2：掩码机制有效性 | 非活跃主体是否被正确排除 | 去除 Transformer 侧结构掩码，观察性能与无效主体干扰 | Full vs `mask_mode=obs_only` |
| 实验 3：动态参与鲁棒性 | 节点加入/退出频率升高时性能是否稳定 | 构造 low/mid/high churn 场景 | low/mid/high churn |

## 3. 实验 1：序列建模有效性

### 3.1 目的

验证将动态参与 P2P 交易建模为多智能体序列决策问题是否有效。重点比较 Transformer 自注意力结构与固定拼接 MLP、共享策略 MAPPO 以及可选变长集合/图结构 baseline 在动态主体集合下的表现差异。实验不仅关注平均性能，还关注 active agent count 变化时各方法的性能波动和退化幅度。

### 3.2 实验假设

- 动态 P2P 交易中，主体之间存在交易报价、供需匹配、储能充放电和碳责任传递等交互关系。
- 自注意力能够根据当前活跃主体集合自适应建模主体间关系，而固定拼接或 MLP 策略更依赖固定主体位置和固定输入结构。
- MLP-Pad 和 MAPPO-Shared 经过 padding、共享策略或 active mask 后可以在动态参与场景中运行，但它们对输入位置、集中 critic 结构或简单聚合机制仍较敏感。
- 当活跃主体集合变化时，DP-LCRL 的性能下降应小于固定拼接、共享 MAPPO 以及可选变长集合/图结构 baseline。

### 3.3 对照方法

| 方法 | 说明 | 对照意义 |
|---|---|---|
| DP-LCRL | Transformer encoder-decoder + self-attention + agent mask | 本文方法，验证“智能体掩码 + 自注意力序列建模”对动态参与主体集合的适应能力 |
| MLP-Pad | 将所有主体 padding 到 `num_agents = 30`，非活跃主体观测置零或填充 mask，但输入仍固定拼接后送入 MLP | 最基础非序列 baseline，用于证明普通固定拼接方法在动态规模下不够稳定 |
| MAPPO-Shared | 使用共享 actor，critic 输入全局状态或拼接状态，并加入 active mask | 更公平的 MAPPO baseline，至少具备跨主体参数复用能力，可检验常规多智能体策略在动态参与下的适应性 |
| DeepSets baseline（可选） | 对每个活跃主体独立编码，再通过 sum/mean/max pooling 聚合为集合表示 | 动态规模 baseline，用于证明 DP-LCRL 不只是因为能处理变长输入才取得优势 |
| GNN baseline（可选） | 将主体视为节点，将交易关系、供需匹配或碳责任传递关系视为边，通过消息传递建模主体关系 | 关系建模 baseline，用于检验 DP-LCRL 与显式图关系建模方法相比是否仍有竞争力 |

如果当前代码中暂时没有完整 baseline，可先实现 MLP-Pad 和 MAPPO-Shared 两个最小可比版本：保持环境、奖励、训练步数、动作空间、最大主体数和随机种子一致，只替换策略网络结构。DeepSets 或 GNN 可作为加强 baseline 纳入；若暂不实现，应在论文中说明其为后续扩展或补充实验方向。

### 3.4 实验设置

- 最大主体数：`num_agents = 30`
- 动态参与范围：`min_agents` 到 `num_agents`
- 训练场景：使用相同训练步数、相同随机种子、相同环境参数。
- 测试场景：在多个 active agent count 下评估，例如 `5, 10, 15, 20, 25, 30`；同时保留固定 30-agent 场景，用于判断 baseline 在固定规模下是否接近 DP-LCRL。
- 每个方法至少运行 3 个随机种子。

### 3.5 关键指标

- `episode_reward`
- `p2p_trade_volume`
- `grid_import`
- `renewable_utilization`
- `carbon_responsibility`
- `performance_drop_under_dynamic_participation`

### 3.6 判定标准

实验支持序列建模有效性需满足：

1. DP-LCRL 在多数动态参与规模下取得更高或更稳定的 reward。
2. 随 active agent count 变化，DP-LCRL 的性能波动小于 MLP-Pad、MAPPO-Shared 以及可选 DeepSets/GNN baseline。
3. DP-LCRL 在 P2P 成交量、可再生能源消纳或碳责任指标上至少有一项表现更优。
4. 若 MLP-Pad 或 MAPPO-Shared 在固定 30-agent 规模下接近 DP-LCRL，但在 active count 变为 `5, 10, 15, 20, 25` 时明显波动或退化，可说明 DP-LCRL 的优势不是单纯网络容量更大，而是“智能体掩码 + 自注意力序列建模”提升了动态参与适应能力。
5. 若 DeepSets baseline 在动态规模下优于 MLP-Pad 但仍弱于 DP-LCRL，可说明 DP-LCRL 的优势不只是来自变长输入处理，还来自主体间交互建模。
6. 若 GNN baseline 接近或低于 DP-LCRL，可进一步说明 DP-LCRL 在不显式手工定义交易图边的情况下，也能通过注意力机制学习有效的主体关系。

### 3.7 建议图表

| 图 | 横轴 | 纵轴 | 曲线 |
|---|---|---|---|
| 跨规模 reward 曲线 | active agent count | episode reward | DP-LCRL, MLP-Pad, MAPPO-Shared, DeepSets/GNN（可选） |
| 交易性能曲线 | active agent count | P2P trade volume | DP-LCRL, MLP-Pad, MAPPO-Shared, DeepSets/GNN（可选） |
| 低碳指标曲线 | active agent count | carbon responsibility 或 renewable utilization | DP-LCRL, MLP-Pad, MAPPO-Shared, DeepSets/GNN（可选） |
| 动态退化曲线 | active agent count | performance drop relative to fixed 30-agent setting | DP-LCRL, MLP-Pad, MAPPO-Shared, DeepSets/GNN（可选） |

## 4. 实验 2：掩码机制有效性

### 4.1 目的

验证智能体掩码是否能够正确排除非活跃主体，使策略只基于当前有效主体集合进行注意力计算、动作生成和价值估计。

### 4.2 实验假设

- 在动态参与场景下，非活跃主体是占位输入，不应参与注意力交互和动作决策。
- Full DP-LCRL 通过智能体掩码屏蔽非活跃主体，能够降低无效主体对策略的干扰。
- `mask_mode=obs_only` 去除 Transformer 侧结构掩码后，非活跃主体仍可能影响注意力和动作生成，因此动态场景性能会下降。

### 4.3 对照方法

| 方法 | mask_mode | 说明 |
|---|---|---|
| Full DP-LCRL | `full` | 启用结构掩码，非活跃主体不参与注意力和动作决策 |
| w/o Agent Mask | `obs_only` | 仅保留观测层面的处理，不在 Transformer 侧屏蔽非活跃主体 |

### 4.4 实验设置

- 保持 CMTM、奖励函数、训练步数和环境参数一致。
- 仅改变 `mask_mode`。
- 在动态参与场景中测试，避免所有主体始终活跃。
- 建议加入 inactive noise stress test：对非活跃主体观测注入噪声，检查 Full 是否比 `obs_only` 更稳。

### 4.5 关键指标

- `episode_reward`
- `attn_leakage`
- `attn_entropy`
- `inactive_action_magnitude`
- `p2p_trade_volume`
- `grid_import`
- `carbon_responsibility`

### 4.6 判定标准

实验支持掩码机制有效性需满足：

1. Full DP-LCRL 的 `attn_leakage` 接近 0。
2. 在 inactive noise stress test 下，Full 的 reward 和交易指标退化小于 `obs_only`。
3. Full 中非活跃主体动作应被有效置零或不参与环境有效决策。
4. `obs_only` 若在动态场景或噪声场景下性能下降，说明结构掩码对排除无效主体干扰是必要的。

### 4.7 建议图表

| 图 | 横轴 | 纵轴 | 曲线 |
|---|---|---|---|
| 掩码消融 reward | training steps 或 active agent count | episode reward | Full, obs_only |
| 注意力泄漏 | training steps | attn_leakage | Full, obs_only |
| 非活跃噪声鲁棒性 | noise level | episode reward | Full, obs_only |

## 5. 实验 3：动态参与鲁棒性

### 5.1 目的

验证节点加入和退出频率升高时，DP-LCRL 是否仍能保持稳定的交易决策能力。

### 5.2 实验假设

- 节点加入和退出越频繁，活跃主体集合和交易关系变化越剧烈，策略学习和泛化难度越高。
- 如果 DP-LCRL 的动态序列建模和智能体掩码机制有效，则在 high churn 场景下性能下降应相对平缓。
- 若方法只适合固定主体集合，则在 churn 增大时 reward、P2P 成交和低碳指标会明显退化。

### 5.3 场景设置

| 场景 | step_churn_prob | 说明 |
|---|---:|---|
| Low churn | 0.05 | 低频加入/退出 |
| Mid churn | 0.20 | 中等频率加入/退出 |
| High churn | 0.50 | 高频加入/退出 |

可根据实际环境稳定性微调概率，但三组场景应保持明显差异。

### 5.4 对照方法

| 方法 | 说明 |
|---|---|
| Full DP-LCRL | 动态序列建模 + 智能体掩码 + 自注意力 |
| w/o Agent Mask | 用于观察高 churn 下非活跃主体干扰 |
| MLP/MAPPO baseline | 用于观察非序列建模方法在高 churn 下是否更易退化 |

### 5.5 关键指标

- `episode_reward`
- `reward_std`
- `p2p_trade_volume`
- `grid_import`
- `renewable_utilization`
- `carbon_responsibility`
- `performance_degradation_from_low_to_high_churn`

### 5.6 判定标准

实验支持动态参与鲁棒性需满足：

1. 从 low churn 到 high churn，Full DP-LCRL 的性能下降幅度小于对照方法。
2. High churn 下 Full DP-LCRL 的 reward 方差更低，说明策略稳定性更好。
3. High churn 下 Full DP-LCRL 仍能维持合理 P2P 成交量和低碳指标。
4. 若去掉掩码或序列建模后高 churn 退化明显，可说明动态场景下“序列建模 + 掩码自注意力”是有效机制。

### 5.7 建议图表

| 图 | 横轴 | 纵轴 | 曲线 |
|---|---|---|---|
| churn 鲁棒性 | churn level | episode reward | Full, obs_only, MLP/MAPPO |
| 性能退化比例 | churn level | degradation ratio | Full, obs_only, MLP/MAPPO |
| 交易稳定性 | churn level | reward std 或 P2P volume std | Full, obs_only, MLP/MAPPO |

## 6. 实验实施建议

### 6.1 最小必要实验组合

如果时间有限，优先完成以下组合：

1. Full DP-LCRL vs `mask_mode=obs_only`
2. low/mid/high churn 三个动态参与强度
3. active agent count sweep

这样即使暂时没有 MLP/MAPPO baseline，也能先证明智能体掩码自注意力机制在动态参与场景下的必要性。

### 6.2 完整实验组合

完整版本建议包含：

1. Full DP-LCRL
2. w/o Agent Mask
3. MLP baseline
4. MAPPO baseline
5. low/mid/high churn
6. active agent count sweep
7. inactive noise stress test

### 6.3 结果组织方式

最终论文中建议按如下逻辑呈现：

1. 先用序列建模实验说明动态 P2P 交易不能简单用固定拼接 MLP 处理。
2. 再用掩码消融说明非活跃主体必须被结构性排除。
3. 最后用 churn 鲁棒性说明该机制在真实动态参与条件下仍然稳定。

## 7. 论文表述建议

可在实验分析中使用如下表述：

```text
实验结果表明，DP-LCRL 在不同活跃主体规模和不同参与变化频率下均保持了更稳定的交易性能。与去除智能体掩码的变体相比，完整方法在非活跃主体存在时能够显著降低无效主体对注意力计算和动作生成的干扰；与非序列建模基线相比，基于自注意力的序列策略能够更好地刻画动态活跃主体之间的交易交互关系。上述结果说明，动态参与 P2P 交易的序列化建模以及智能体掩码自注意力机制是提升可变主体集合适应能力的关键因素。
```
