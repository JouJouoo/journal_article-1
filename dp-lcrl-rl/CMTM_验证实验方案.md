# CMTM 验证实验方案

## 1. 实验背景

当前消融结果中，部分消融设置的效果优于完整方法。为避免只通过最终收益指标解释模型有效性，需要补充更具机制性的验证实验，直接检查 CMTM、动态智能体掩码与碳责任流追踪是否按设计工作。

本方案包含 4 个 CMTM 机制验证实验：

1. 储能跨时段碳继承验证
2. P2P 碳流链式传递验证
3. 动态参与下 CMTM 冻结-恢复验证
4. 碳不平衡分解

储能跨时段碳继承是 CMTM 的基础验证；P2P 链式传递、动态参与冻结-恢复和碳不平衡分解是在此基础上的扩展机制验证。

## 2. 总体目标

### 2.1 核心问题

- CMTM 是否能记录储能历史碳含量，并在跨时段放电时继承该历史碳状态，而不是将储能放电视为无记忆、无来源的即时能量？
- P2P 交易中的碳标签是否能沿交易关系正确传递？
- 智能体退出后，储能碳含量是否被冻结，并在重新参与时恢复？
- 脚本化场景中的碳输入、碳输出和残余碳是否能闭合解释？

### 2.2 方法原则

- 优先使用可控脚本化场景，而不是完整训练结果。
- 每个实验只验证一个机制，避免多个机制混在一起。
- 同时记录 full CMTM 与 stateless 版本，形成机制对照。
- 输出 JSON/CSV 结果，便于后续画图和写论文。

## 3. 创新点一：CMTM — 碳记忆追踪模型

本组实验用于验证论文的第一个核心创新点：CMTM 能够在储能跨时段使用、P2P 交易链式传递、动态参与退出与恢复等场景中保持可追踪、可解释的碳责任记录。

| 实验 | 验证内容 | 方法 | 状态 | 对应代码 |
|---|---|---|---|---|
| 脚本化“先充后放” | 储能跨时段碳继承 | full CMTM vs No Pool（同策略反事实） | ✅ 代码就绪 | `eval_cmtm_memory_validation.py` |
| P2P 碳流链式 | 碳标签沿交易链传递 | 两卖家（低碳+高碳）→ 买家 | ✅ 已运行 | 实验1 in `eval_cmtm_extended.py` |
| 动态参与冻结-恢复 | 节点退出期碳质量不变，回归后正确恢复 | Phase 1 充电 → Phase 2 退出 → Phase 3 放电 | ✅ 已运行 | 实验2 in `eval_cmtm_extended.py` |
| 碳不平衡分解 | 解释 ~9.4% 残差（= 1 - η_ch × η_dis） | 追踪储能碳池 input = output + residual | ✅ 已运行 | 实验3 in `eval_cmtm_extended.py` |

## 4. 实验 1：储能跨时段碳继承验证

### 4.1 目的

验证 CMTM 是否能在“先充电、后放电”的跨时段场景中正确继承储能碳含量。该实验是 CMTM 的最基础机制验证：如果储能碳池不能跨时段保存碳质量，后续 P2P 碳标签传递和动态参与恢复都缺少可信基础。

### 4.2 实验假设

- full CMTM 会在充电时把输入电量对应的碳质量写入储能碳池。
- 后续放电时，释放电量应携带此前存入储能池的历史碳强度。
- No Pool 或 stateless 反事实版本缺少储能碳池记忆，因此无法正确反映“过去充入的碳”在未来放电中的继承关系。

### 4.3 场景设置

采用脚本化“先充后放”场景：

| 阶段 | 设置 | 预期行为 |
|---|---|---|
| 充电阶段 | 节点从电网或指定碳源充电 | `carbon_mass` 增加，`storage_intensity` 更新 |
| 静置阶段 | 不充不放 | 储能碳质量保持稳定 |
| 放电阶段 | 节点放电供负载、P2P 或电网输出 | 输出碳强度继承储能历史碳强度 |

### 4.4 对照组

| 组别 | 说明 |
|---|---|
| Full CMTM | 启用储能碳含量池，记录 `carbon_mass` 与 `storage_intensity` |
| No Pool | 同策略反事实，不使用储能碳池记忆 |

### 4.5 关键指标

- `storage_charge_energy`
- `storage_charge_carbon`
- `carbon_mass_after_charge`
- `storage_intensity_after_charge`
- `discharge_energy`
- `storage_discharge_carbon`
- `discharge_carbon_intensity`
- `full_vs_no_pool_discharge_carbon_delta`

### 4.6 判定标准

实验通过需满足：

1. Full CMTM 中，充电后 `carbon_mass` 随输入碳增加。
2. 静置阶段 `carbon_mass` 不应无故变化。
3. 放电阶段 `discharge_carbon_intensity` 应接近充电后形成的 `storage_intensity`。
4. Full CMTM 与 No Pool 在放电碳责任上存在可解释差异。
5. 若考虑充放电效率，碳恢复率应与 `η_ch × η_dis` 一致，剩余部分进入残余分解实验解释。

### 4.7 对应代码

该实验对应已有脚本：

```text
dp_lcrl_rl/scripts/eval/eval_cmtm_memory_validation.py
```

建议输出到：

```text
reports/cmtm_memory_validation/
```

## 5. 实验 2：P2P 碳流链式传递验证

### 5.1 目的

验证 CMTM 的 P2P 碳标签能沿交易链正确传递。重点检查购电节点从多个不同碳来源购电时，其购电碳足迹是否等于分来源加权平均。

### 5.2 实验假设

- full CMTM 会保留储能历史碳强度，因此售电节点的放电碳标签受历史充电来源影响。
- stateless 版本缺少储能碳含量池记忆，其放电碳标签更多依赖当前时刻的即时碳计算。
- 当一个购电节点同时从低碳售电节点和高碳售电节点购电时，full CMTM 的购电碳强度应等于两路 P2P 碳流的加权平均。

### 5.3 场景设置

使用 3 个节点：

- Seller A：低碳来源节点，主要由光伏充电，储能碳强度较低。
- Seller B：高碳来源节点，主要由电网充电，储能碳强度较高。
- Buyer C：购电节点，同时从 Seller A 和 Seller B 购电。

建议使用固定动作脚本，避免策略噪声影响验证：

- Seller A：放电并售电。
- Seller B：放电并售电。
- Buyer C：提交购电需求，报价高于两个售电节点。

### 5.4 对照组

| 组别 | cmtm_mode | 说明 |
|---|---|---|
| Full CMTM | `full` | 启用储能碳含量池与历史碳记忆 |
| Stateless | `stateless` | 不保留储能碳历史记忆 |

### 5.5 关键指标

- `p2p_carbon_label_a_to_c`
- `p2p_carbon_label_b_to_c`
- `buyer_p2p_import_carbon`
- `buyer_p2p_import_energy`
- `buyer_weighted_p2p_carbon_intensity`
- `buyer_load_carbon`
- `full_vs_stateless_load_carbon_delta`

### 5.6 判定标准

实验通过需满足：

1. Full CMTM 中，Seller A 和 Seller B 的 P2P 碳标签不同。
2. Buyer C 的 P2P 购电碳强度等于两路来源的能量加权平均。
3. full 与 stateless 的购电碳标签或负载碳责任存在可解释差异。
4. 输出结果中能追溯每一笔 P2P 交易的能量、价格、碳标签和碳责任。

## 6. 实验 3：动态参与下 CMTM 冻结-恢复验证

### 6.1 目的

验证节点退出期间，储能碳含量池不会被错误更新；节点重新活跃后，其放电碳强度应恢复为退出前保存的历史值。

### 6.2 实验假设

- 当 `agent_mask=0` 时，该节点不参与交易、充放电和碳责任更新。
- 退出期间，节点的 `carbon_mass` 与 `storage_intensity` 应保持不变。
- 节点重新活跃并放电时，输出碳强度应反映退出前冻结的储能碳历史。

### 6.3 阶段设计

| 阶段 | 时间步 | 设置 | 预期行为 |
|---|---:|---|---|
| Phase 1 | 0-4 | 全部节点活跃，目标节点充电 | 目标节点储能碳质量增加 |
| Phase 2 | 5-12 | 目标节点退出，`agent_mask=0` | 目标节点储能碳质量冻结 |
| Phase 3 | 13-20 | 目标节点恢复活跃并放电 | 放电碳强度匹配冻结前历史值 |

### 6.4 关键指标

- `carbon_mass_before_exit`
- `carbon_mass_during_inactive`
- `carbon_mass_after_reactivation`
- `storage_intensity_before_exit`
- `discharge_carbon_intensity_after_reactivation`
- `carbon_mass_freeze_error`

### 6.5 判定标准

实验通过需满足：

1. Phase 2 期间，退出节点的 `carbon_mass` 不发生变化。
2. Phase 2 期间，退出节点不产生 P2P 交易、负载碳责任或储能碳更新。
3. Phase 3 恢复后，节点放电碳强度接近退出前的储能碳强度。

## 7. 实验 4：碳不平衡分解

### 7.1 目的

解释脚本化“先充后放”场景中，碳恢复率剩余约 10% 的去向，验证碳输入、输出和残余储能碳之间是否闭合。

### 7.2 实验假设

脚本化场景中的碳输入不会全部在同一阶段释放，其中一部分可能保留在储能中，另一部分通过负载消耗或电网输出离开系统。

### 7.3 分解项

追踪以下三项输出：

- `final_storage_carbon`
- `grid_export_carbon`
- `load_carbon`

同时记录输入项：

- `total_grid_import_carbon`
- `total_p2p_import_carbon`
- `total_storage_charge_carbon`

### 7.4 守恒关系

核心检查公式：

```text
total_input = total_output + residual
```

其中：

```text
total_output = grid_export_carbon + load_carbon
residual = final_storage_carbon
```

### 7.5 判定标准

实验通过需满足：

1. `total_input - total_output - residual` 接近 0。
2. 剩余约 10% 的碳能被明确分解到 `final_storage_carbon`、`grid_export_carbon` 或 `load_carbon`。
3. full CMTM 下的碳守恒误差显著小于无记忆或弱追踪版本。

## 8. 实现顺序

### Step 1：确认实验 1

优先确认脚本化“先充后放”验证入口：

```text
dp_lcrl_rl/scripts/eval/eval_cmtm_memory_validation.py
```

脚本输出：

- `reports/cmtm_memory_validation/results.json`
- `reports/cmtm_memory_validation/results.csv`
- 可选：`reports/cmtm_memory_validation/summary.md`

### Step 2：实现实验 2

新增 P2P 链式传递验证入口，例如：

```text
dp_lcrl_rl/scripts/eval/eval_cmtm_p2p_chain_validation.py
```

脚本输出：

- `reports/cmtm_p2p_chain_validation/results.json`
- `reports/cmtm_p2p_chain_validation/results.csv`
- 可选：`reports/cmtm_p2p_chain_validation/summary.md`

### Step 3：实现实验 3

新增冻结-恢复验证脚本，例如：

```text
dp_lcrl_rl/scripts/eval/eval_cmtm_freeze_recovery.py
```

### Step 4：实现实验 4

新增碳不平衡分解脚本，例如：

```text
dp_lcrl_rl/scripts/eval/eval_cmtm_carbon_balance_decomposition.py
```

## 9. 建议输出表格

### 9.1 实验 1 输出表

| mode | phase | charge_carbon | carbon_mass | storage_intensity | discharge_carbon | recovery_rate |
|---|---|---:|---:|---:|---:|---:|
| full | charge |  |  |  |  |  |
| full | discharge |  |  |  |  |  |
| no_pool | charge |  |  |  |  |  |
| no_pool | discharge |  |  |  |  |  |

### 9.2 实验 2 输出表

| mode | seller | buyer | energy | carbon_label | carbon_responsibility |
|---|---|---|---:|---:|---:|
| full | A | C |  |  |  |
| full | B | C |  |  |  |
| stateless | A | C |  |  |  |
| stateless | B | C |  |  |  |

### 9.3 实验 3 输出表

| phase | step | agent_id | active | carbon_mass | storage_intensity | discharge_intensity |
|---|---:|---:|---|---:|---:|---:|
| Phase 1 |  |  |  |  |  |  |
| Phase 2 |  |  |  |  |  |  |
| Phase 3 |  |  |  |  |  |  |

### 9.4 实验 4 输出表

| mode | total_input | grid_export_carbon | load_carbon | final_storage_carbon | balance_error |
|---|---:|---:|---:|---:|---:|
| full |  |  |  |  |  |
| stateless |  |  |  |  |  |

## 10. 论文写作用途

这 4 个实验不直接替代训练消融，而是作为机制验证补充：

- 实验 1 支撑“储能碳状态可跨时段继承”。
- 实验 2 支撑“P2P 交易碳标签可追踪”。
- 实验 3 支撑“动态参与智能体下储能碳状态可保持”。
- 实验 4 支撑“碳责任流满足可解释的输入-输出闭合”。

如果完整方法最终收益不总是最高，可以用这些实验说明：完整方法的贡献不只体现在 reward 最大化，还体现在碳责任追踪的物理一致性、动态参与鲁棒性和可解释性。
