# DATA_DICTIONARY.md — 数据字典与数据结构

## 1. 外部数据集（候选，已完成元数据核验）

- 名称：User Perception of Generative AI in Mobile Applications: Google Play Store Reviews Dataset
- DOI：10.17632/dnkdxxdpzk.1　版本：V1　发布：2026-10-22
- 作者：Üstün Özen（Atatürk University）、Fatih Demir
- 许可：**CC BY 4.0**（允许共享与改编，须署名）
- 文件：`reviews_all_GenAI.csv`，CSV，UTF-8，约 25.7 MB（页面所列，**实际文件未下载**）
- 规模：每应用 10,000 条，共 7 个应用，合计 70,000 条
- 采集方式：`google-play-scraper`，参数 `lang='en'`、`country='us'`
- 覆盖应用：ChatGPT(OpenAI)、Google Gemini、Microsoft Copilot、Perplexity AI、Claude AI(Anthropic)、DeepSeek AI、Grok
- 研究用途：Mobile HCI、UX、信任、可用性障碍；作者使用 BERTopic + VADER

### 1.1 字段清单（依据数据集页面描述，字段名与类型待下载后实测确认）

| 字段 | 说明 | 用途 | 状态 |
|---|---|---|---|
| `package_name` | Play 包名 | 应用识别（唯一键辅助） | 待实测 |
| `review_id` | 评论唯一 ID | **追溯主键** | 待实测 |
| `user_name` | 用户昵称 | **不展示、不入分析**（隐私） | 待实测 |
| `user_image` | 头像 URL | **不使用**（隐私） | 待实测 |
| `content` | 评论文本 | 分析主输入 | 待实测 |
| `score` | 1–5 星 | 评分信号 | 待实测 |
| `thumbs_up_count` | 有用数 | 可选权重 | 待实测 |
| `review_created_version` | 评论时 App 版本 | 时间/版本分析 | 待实测 |
| `app_version` | 抓取时最新版本 | 参考 | 待实测 |
| `at_utc` | 评论时间（UTC） | 时间趋势 | 待实测（注意页面提到"localized Istanbul time"，时区需实测） |
| `reply_content` | 开发者回复文本 | 可选（官方响应） | 待实测 |
| `replied_at_utc` | 回复时间 | 可选 | 待实测 |
| `scrape_lang` | 抓取语言（en） | 语言核验 | 待实测 |
| `scrape_country` | 抓取国家（us） | 地域核验 | 待实测 |
| `scraped_at_utc` | 抓取时间 | **决定数据新鲜度** | 待实测 |
| `app_name` | 应用标签 | 应用识别主键 | 待实测 |

### 1.2 结论

**部分可用 → 实际可用度高**：应用覆盖与字段结构满足研究问题（含 ChatGPT/Gemini/Perplexity），样本充足（10k/应用）。主要限制：
1. 仅美国区、仅英文——不代表中文用户；
2. 抓取时间点未知（`scraped_at_utc` 待实测），时间趋势可能集中在短窗口；
3. 可能存在重复/灌水/非英文残留，需质量清洗；
4. `user_name`/`user_image` 属个人数据，必须脱敏处理。

### 1.3 抽样规则（阶段 1 执行）

- 目标样本：每产品 150 条，共 450 条（ChatGPT / Gemini / Perplexity）。
- 方法：**分层随机抽样**，分层变量 = `score`（1–5 五层），`random_state=42`。
- 记录抽样清单到 `data/samples/sample_450.csv`（含 `review_id`），保证可复现。
- 若某产品某星级不足 30 条 → 记录实际数量，按真实覆盖降级，**不补造数据**。
- 抽样后再次核对：去重、语言抽样检测（ASCII 比例 + 简单语言判别）、日期范围统计。

## 2. 内部数据库（SQLite，`data/feedbacklens.db`）

### 2.1 `projects`
| 列 | 类型 | 说明 |
|---|---|---|
| id | INTEGER PK | |
| name | TEXT NOT NULL | 项目名 |
| description | TEXT | |
| status | TEXT | draft/analyzing/analyzed/archived |
| created_at | TEXT | UTC ISO |
| updated_at | TEXT | UTC ISO |
| deleted_at | TEXT | 软删除 |

### 2.2 `products`
| 列 | 类型 | 说明 |
|---|---|---|
| id | INTEGER PK | |
| project_id | INTEGER FK | |
| name | TEXT | 产品名（ChatGPT/Gemini/Perplexity） |
| package_name | TEXT | Play 包名（可选） |

### 2.3 `import_batches`
| 列 | 类型 | 说明 |
|---|---|---|
| id | TEXT PK | UUID |
| project_id | INTEGER | |
| filename | TEXT | |
| row_count | INTEGER | |
| accepted_count | INTEGER | |
| rejected_count | INTEGER | |
| quality_report_json | TEXT | 质量报告快照 |
| created_at | TEXT | |

### 2.4 `feedback`
| 列 | 类型 | 说明 |
|---|---|---|
| id | INTEGER PK | 内部主键 |
| project_id | INTEGER | |
| batch_id | TEXT | 导入批次 |
| review_id | TEXT | 外部唯一 ID |
| product_id | INTEGER | |
| content | TEXT NOT NULL | **原文，不可修改** |
| score | INTEGER | 1–5，可空 |
| review_at | TEXT | UTC，可空 |
| lang | TEXT | 检测语言 |
| source | TEXT | mendeley / manual |
| meta_json | TEXT | 其他原始字段 |
| content_hash | TEXT | 去重 |
| created_at | TEXT | |
| 唯一约束 | | (project_id, review_id) |

### 2.5 `analyses`
| 列 | 类型 | 说明 |
|---|---|---|
| id | INTEGER PK | |
| feedback_id | INTEGER FK | |
| run_id | TEXT | 分析批次 |
| model_name | TEXT | 如 llama3.1:8b |
| model_version | TEXT | |
| prompt_version | TEXT | 如 v1.0 |
| themes_json | TEXT | JSON 数组 |
| sentiment | TEXT | positive/neutral/mixed/negative |
| intent | TEXT | praise/complaint/feature_request/question/other |
| pain_point | TEXT | |
| feature_request | TEXT | |
| severity | INTEGER | 1–5 |
| evidence_quote | TEXT | 原文子串 |
| confidence | REAL | 0–1 |
| needs_human_review | INTEGER | 0/1 |
| is_valid | INTEGER | quote 校验结果 |
| raw_output | TEXT | 模型原始输出 |
| error | TEXT | |
| status | TEXT | pending/running/done/failed |
| created_at | TEXT | |
| 幂等 | | UNIQUE(feedback_id, model_name, prompt_version) |

### 2.6 `issues`
| 列 | 类型 | 说明 |
|---|---|---|
| id | INTEGER PK | |
| project_id | INTEGER | |
| title | TEXT | 问题陈述 |
| theme | TEXT | 主题 |
| evidence_ids_json | TEXT | feedback_id 数组 |
| evidence_count | INTEGER | |
| products_json | TEXT | 涉及产品 |
| status | TEXT | candidate/confirmed/dismissed |
| created_at | TEXT | |

### 2.7 `priorities`
| 列 | 类型 | 说明 |
|---|---|---|
| id | INTEGER PK | |
| project_id | INTEGER | |
| issue_id | INTEGER | |
| freq_score | REAL | 0–5 |
| severity_score | REAL | 0–5 |
| impact_score | REAL | 0–5（PM 可改） |
| scope_score | REAL | 0–5 |
| cost_score | REAL | 0–5（PM 输入，可空） |
| total_score | REAL | 加权结果 |
| ai_tier | TEXT | P1/P2/P3 |
| pm_override_tier | TEXT | 可空 |
| rationale | TEXT | 自动生成理由 |
| pm_note | TEXT | |
| updated_at | TEXT | |

### 2.8 `actions`
| 列 | 类型 | 说明 |
|---|---|---|
| id | INTEGER PK | |
| project_id | INTEGER | |
| issue_id | INTEGER | |
| proposal | TEXT | 建议措施 |
| expected_effect | TEXT | |
| assumptions | TEXT | |
| uncertainty | TEXT | |
| status | TEXT | ai_draft/pm_confirmed/pm_rejected |
| decided_by | TEXT | |
| decided_at | TEXT | |
| reject_reason | TEXT | |

### 2.9 `eval_runs` / `eval_labels`
| 表 | 关键列 |
|---|---|
| eval_runs | id, run_id, sample_size, labeler, metrics_json, confusion_json, model_name, created_at |
| eval_labels | id, feedback_id, labeler, gold_themes_json, gold_sentiment, gold_intent, gold_severity, notes, labeled_at |

## 3. 枚举与取值约定

- sentiment：`positive | neutral | mixed | negative`
- intent：`praise | complaint | feature_request | question | other`
- severity：1 轻微 / 2 轻微但重复 / 3 明显不便 / 4 严重影响 / 5 阻断使用或流失
- 主题词表（v1 草案，**需在真实数据上先做主题发现后冻结**）：账号与登录、价格与订阅、回答质量与准确性、速度与延迟、稳定性与崩溃、功能缺失、界面与易用性、隐私与数据安全、客服与支持、跨平台/同步、模型能力边界（幻觉/拒答）、其他
- 优先级：P1 ≥ 4.0；2.5 ≤ P2 < 4.0；P3 < 2.5

## 4. 数据质量指标（导入时必须计算）

1. 行数 / 去重后行数 / 重复率
2. 每产品样本量
3. `content` 缺失率、平均/中位长度、<5 字符占比
4. `score` 缺失率、分布（1–5）、越界数
5. 日期解析成功率、最小/最大日期、按周分布是否断层
6. 语言分布（ASCII 占比启发式 + 简单判别，非精确）
7. 抽样偏差检查：抽样前后 `score` 分布对比（卡方仅作提示，不作结论）

## 5. 隐私与合规

- `user_name`、`user_image` **不入库**（导入时直接丢弃）。
- 报告中不出现任何用户昵称；仅用 `review_id` 追溯。
- 引用原文时保留原样（学术研究引用范畴），但避免引用含个人信息的内容（导入时过滤邮箱/电话/URL 模式并标记）。
- 数据集署名：`Özen, Ü. & Demir, F. (2026). ... Mendeley Data, V1, doi:10.17632/dnkdxxdpzk.1, CC BY 4.0`。
