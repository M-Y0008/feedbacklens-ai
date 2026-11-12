# FeedbackLens AI

把应用商店评论变成**可追溯到原文**的产品决策输入。

FeedbackLens AI 是一个面向产品经理的 Voice-of-Customer 洞察工具：导入非结构化的用户评论，
自动抽取主题、情感、意图、严重度与证据引文，再聚合成痛点、优先级与改进建议。
每一条洞察都能回溯到具体的 `feedback_id` 与原文片段，不做无证据的推断。

## 它解决什么问题

- 评论太多，人工读不完：450 条评论人工整理一次要数小时，且容易漏掉低频但严重的问题。
- 通用大模型直接总结不可信：没有原文引用、无法核对、无法复现。
- 决策需要证据链：哪个主题有多少条支持、来自哪个产品、严重度分布如何，都要能查。

## 设计原则

1. **证据优先**：AI 输出必须带原文引文，且做严格子串校验；不通过则标记为需要人工复核。
2. **事实与解释分离**：原始评论永不覆盖，AI 解释写入独立字段。
3. **不确定就标注**：证据不足时显式写"证据不足，不作为结论"，而不是给一个看似确定的答案。
4. **零付费依赖**：默认本地 Ollama 推理，SQLite 存储，无云端账号、无 API 费用。

## 架构一句话

Streamlit 界面 → `core/` 领域层（分析、批量、聚合、评分、报告）→ SQLite；
模型通过可插拔后端接入，默认本地 Ollama，也可选任意 OpenAI 兼容接口。

```
app/        Streamlit 界面（9 个页面：上传、分析、洞察、优先级、行动、对比、评估、导出、工作区）
core/       领域逻辑，不依赖 UI（analysis / batch / llm / db / ingest / aggregate / scoring / report）
config/     analysis.json：模型、提示词版本、主题词表、批量与校验参数
data/       SQLite 数据库、样例数据、上传与导出目录（运行产物不入库）
docs/       安装使用说明等
scripts/    数据构建与评估脚本
```

## 快速开始

前置：Python 3.12、[Ollama](https://ollama.com/) 已安装。

```powershell
# 1) 依赖
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2) 本地模型（约 5.2 GB，只需一次）
ollama pull qwen3:8b

# 3) 启动
streamlit run app/main.py
# 浏览器打开 http://localhost:8501
```

首次启动会自动创建/迁移数据库。没有显卡也能跑（Ollama 回退 CPU），但分析会明显变慢。

完整的环境要求、数据导入、页面说明与常见问题见 [docs/INSTALL_AND_USAGE.md](docs/INSTALL_AND_USAGE.md)。

## 数据

- 数据来源：Google Play 上 GenAI 类应用的公开评论，研究数据集
  **Özen, Ü. & Demir, F. (2026), Mendeley Data, V1, doi:10.17632/dnkdxxdpzk.1**。
- 许可：**CC BY 4.0**。再分发或改编时必须保留上述署名与 DOI。
- 原始 CSV **不随仓库分发**：其中包含评论者邮箱与头像 URL 等个人信息，不适合公开。
  请从上述 DOI 自行下载，放到 `data/raw/`，再运行 `python scripts/p1_build_dataset.py`。
- `data/samples/sample_450.csv` 是已脱敏的 450 条样例数据，可直接用于试跑。
- 详细说明与字段定义见 [data/README.md](data/README.md) 与 [DATA_DICTIONARY.md](DATA_DICTIONARY.md)。

## 评估结果（60 条人工标注，2026-11）

模型：本地 `qwen3:8b`，提示词版本 `v2.2-p2`。同一批 60 条评论与人工标注逐条对比：

| 指标 | 结果 |
|---|---|
| 主题命中率（至少一个标签重合） | 81.7% |
| 主题集合完全一致 | 75.0% |
| 情感准确率 | 83.3% |
| 意图准确率 | 86.7% |
| 严重度误差 ≤1 | 86.7% |
| 四字段全对（主题+情感+意图+严重度） | 50.0% |
| 证据引文可在原文中严格命中 | 100% |

**这是 60 条的小样本结果**：单人标注、每产品固定 20 条、无标注者间一致性检验，
不能外推到全量评论或其他数据集。多数主题支持数小于 5，单条改动即可让指标跳动数个点。

## 已知限制

- 数据为美国区英文评论，界面与报告为中文；结论不能直接外推到其他地区/语言。
- 主题词表中 `other` 是"证据不足"兜底桶，会吸收一部分真实主题（当前召回偏低）。
- 严重度完全一致率约 61.7%，严重度是四字段中最不稳定的一项。
- 未做标注者间一致性（Kappa），评估数字的置信区间较宽。

## 可选：在线模型（默认关闭）

为了对比本地模型与在线模型的效果，`core/llm.py` 支持任意 OpenAI 兼容接口。
默认关闭，需要显式设置环境变量（API key 只放环境变量，禁止写入仓库）：

```powershell
$env:FEEDBACKLENS_API_BASE="https://api.groq.com/openai/v1"
$env:FEEDBACKLENS_API_MODEL="openai/gpt-oss-120b"
$env:FEEDBACKLENS_API_KEY="<你的 key>"
python scripts/p2_online_compare.py --probe
```

不设置这些变量时，应用始终使用本地 Ollama，不会产生任何费用。

## 许可

- **数据**：CC BY 4.0，署名见上（Özen & Demir, doi:10.17632/dnkdxxdpzk.1）。
- **代码**：作者尚未最终指定开源许可，正式发布前会在 `LICENSE` 中补充
  （建议 MIT 或 Apache-2.0）。
