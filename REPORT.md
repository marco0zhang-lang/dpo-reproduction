# DPO 技术复现报告

## 1. 目标与论文
Rafailov 等（2023）将带 KL 约束的奖励最大化转化为直接偏好分类。本文复现式 (7)，检验冻结参考模型条件下偏好对数比是否增加。论文原始实验涉及情感控制、摘要、单轮对话；本实验不复现这些 benchmark 的具体分数。

## 2. 数据与环境
合成词表大小 12，prompt 由两个 token 组成，分别来自 0–3 与 4–7。chosen 为 [8,10]，rejected 为 [9,11]。固定 seed 42 划分 12 个训练组合和 4 个测试组合，无组合重叠，但共享 token 与统一偏好规则。GRU embedding 24、hidden 32；先用两种回答等权做 25 步 SFT 形成中性参考模型，再复制参数并冻结参考模型。CPU，PyTorch 2.10.0+cpu。Adam，预训练学习率 0.01，偏好训练学习率 0.003，全批量，每 epoch 一步，30 epoch，种子 42/43/44，beta=0.1。无外部数据下载。

## 3. 实现过程
对 prompt+response 做自回归前向；通过 gather 获取真实回答 token 的 log probability 并求和；排除 prompt；计算 policy 与 reference 的 chosen/rejected 对数概率差；以负 log sigmoid 构建 DPO 损失；反向传播只更新 policy。chosen-SFT 对照使用相同初始化、学习率与更新次数。每轮在 held-out 组合上评估。测试集不用于梯度更新；没有独立验证集，因此不做最优超参数选择。

## 4. 指标及结果
偏好准确率定义为 policy 对 chosen 的 log probability 大于 rejected 的比例；relative margin 为 policy log ratio 减去 reference log ratio；DPO loss 为测试集上的目标函数。三种子均值：初始准确率 41.67%，初始 loss 0.693147，初始相对 margin 约 0。DPO 最终准确率 100%，loss 0.339224，相对 margin 9.099170；chosen-SFT 最终准确率 100%，loss 0.513643，相对 margin 3.987142。曲线阴影为种子间总体标准差，不是置信区间。详细逐轮数据见 results/metrics.csv。beta=0.5 敏感性实验见 results/beta05/summary.json；不同 beta 的 loss 不宜直接横向比较，因为目标尺度不同。

## 5. 与原论文比较
一致点：无需显式奖励模型与在线 RL；使用固定参考策略；成对偏好损失能够提升偏好相对 margin。差异：论文采用真实文本和预训练语言模型，报告任务级指标与人类/模型偏好评价；这里采用小 GRU 与合成 token，不能把 100% 准确率与论文胜率直接比较。SFT 同样达到 100%，说明任务存在天花板效应，DPO 的更大 margin 仅反映本任务上的优化行为，不等于更好的生成质量。

## 6. 可靠性与改进
四项单元测试验证损失初值 ln2、梯度方向、数据组合隔离和输出形状。三随机种子降低单次初始化偶然性。后续应引入真实偏好数据、prompt-dependent 偏好、独立验证集、预训练小模型及生成质量评估，才有可能进一步接近原论文。当前未运行真实文本实验，未执行 GitHub 远端 CI，也没有 GPU 结果。

## 7. 复现步骤与证据
参照 README 安装依赖、运行测试、训练和 beta 实验。results/run.log、tests.log、summary.json、metrics.csv 和 experiment.png 均为本次本地实际运行产物。建议提交时附原始 CSV、曲线、代码仓库 commit hash，并在自己的机器重新运行后截取终端及 GitHub Actions 页面。当前图是程序生成的结果图，不是伪造的终端截图。
