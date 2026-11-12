# FeedbackLens AI — 数据说明与许可

## 1. 许可（必须署名）

本项目使用的评论数据来自公开研究数据集：

> Özen, Ü. & Demir, F. (2026). *Google Play GenAI Applications Reviews*.
> Mendeley Data, V1. doi:10.17632/dnkdxxdpzk.1

该数据集以 **CC BY 4.0** 许可发布。任何再分发、改编或在衍生作品中使用，都必须：

1. 保留上述作者、标题、DOI 与许可说明；
2. 注明是否做过修改（本项目做了抽样、字段裁剪与去重）；
3. 附上许可全文链接：https://creativecommons.org/licenses/by/4.0/

## 2. 仓库里有什么

| 路径 | 内容 | 是否入库 |
|---|---|---|
| `data/samples/sample_450.csv` | 已脱敏的 450 条样例评论（本项目实际使用的样本） | 是 |
| `data/raw/` | 原始下载 CSV / ZIP | **否**（见第 3 节） |
| `data/feedbacklens.db` | SQLite 数据库（评论、分析、标注） | 视发布配置而定 |
| `data/uploads/` | 用户上传文件的副本 | 否（运行时产物） |
| `data/exports/` | 导出文件 | 否（运行时产物） |

## 3. 为什么原始 CSV 不随仓库分发

原始 `reviews_all_GenAI.csv` 包含评论者邮箱、昵称与头像 URL 等个人信息。
即使数据集整体为 CC BY 4.0，公开再分发这些个人信息也不合适，因此：

- 请自行从 DOI 页面下载原始数据；
- 放到 `data/raw/reviews_all_GenAI.csv`；
- 运行 `python scripts/p1_build_dataset.py` 重新构建样例数据与数据库。

应用与数据库本身不保存 `user_name` / `user_image`，只保留 `review_id` 用于追溯。

## 4. 抽样与已知偏差

- 每产品固定 20 条标注、共 60 条，按 `feedback_id` 顺序抽取，**不是随机抽样**。
- 原始数据以美国区、英文评论为主，存在地区与语言偏差。
- 评论时间窗有限，不能代表产品全生命周期。
- 任何基于本项目样本的结论都应带上上述限制。
