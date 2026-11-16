# 训练目录规划：一包一树

原则：**每上传一批训练数据，就建一个独立训练包目录**；包与包之间互不共享 `data/`、`runs/`、`saves/`、`models/`。找某次训练的产物时，只进对应包目录即可，不必在挂载盘根下翻全局 `models/` / `saves/`。

轻量模板在 `/cloud/cloud-ssd1/goldthread-train/`（约十几 MB，无模型；说明见该目录 `TEMPLATE.md`）。完整产物包示例：`packages/goldthread-20261106/`。

## 总览

```
/cloud/cloud-ssd1/                          # 挂载盘：成果与训练包（可整盘挂到别的机器）
├── goldthread-train/                      # 轻量模板（复制它建新包）
└── packages/                              # 历次上传对应的独立训练包
    ├── goldthread-20261106/               # 含模型的完整包示例
    ├── <包名-A>/
    └── <包名-B>/

/data/work/<包名>/                         # 本地盘：仅中间数据（过滤数据、checkpoint）
/model/...                                 # 只读共享：底座，各包共用，不复制
```

> 注意：本机若存在坏链 `/data/goldthread`（曾指向已删除的交付目录），不要再用它做 `work_dir`；统一用 `/data/work/<包名>`。

存储分工不变：

| 存储 | 用途 |
| --- | --- |
| `/model` | 底座（各包只读引用） |
| `/data` | 本机中间数据，换机器可丢 |
| `/cloud/cloud-ssd1` | 训练包 + adapter + 合并模型 + 运行记录 |

## 包名怎么起

格式：`<数据别名>-<YYYYMMDD>`，必要时加短后缀区分同日多次上传。

| 示例 | 含义 |
| --- | --- |
| `goldthread-20261106` | 11 月 6 日上传的 GoldThread 数据 |
| `goldthread-20261109-v2` | 同日第二版数据 |
| `goldthread-bc1d8bf6` | 用交付包 config_hash 命名（见 `data/MANIFEST.json`） |

要求：小写、数字、连字符；一个包名对应一棵目录树，不要复用旧包目录覆盖数据。

## 单个训练包目录树

```
packages/<包名>/
├── autotrain.py / setup_env.sh / settings.yaml / AGENTS.md / README.md
├── configs/                 # 方案与基础 YAML（可按本批数据长度调整）
├── tools/                   # 训练工具（与模板一致即可）
├── issues/                  # 本包问题记录；known_issues.json 建议从模板继承
├── docs/                    # 文档副本或指向模板的说明
│
├── data/                    # ★ 本次上传的训练数据（只读）
│   ├── MANIFEST.json        # 来源、条数、sha256
│   ├── dataset_info.json
│   └── *.jsonl
│
├── runs/                    # ★ 本包每次运行的配置、日志、汇总
│   ├── history.jsonl
│   ├── <阶段>.out
│   └── <运行名>/            # 如 20261106-040350-dpo
│       ├── attempt-N.yaml / attempt-N.log
│       ├── run.json / status.json
│       ├── filter_summary_<cutoff>.json
│       ├── merge.log / smoke.log
│       └── eval_results/    # 仅 eval
│
├── saves/                   # ★ 本包最终 LoRA adapter（不含 checkpoint）
│   └── <运行名>/
│
└── models/                  # ★ 本包合并整模（部署从这里取）
    └── <merged_prefix>-<阶段>-<运行名>/
```

同一次运行在包内三处共用**同一个运行名** `<日期>-<时间>-<阶段>`，例如 `20261106-040350-dpo`：

| 内容 | 包内路径 |
| --- | --- |
| 运行记录 | `runs/<运行名>/` |
| 最终 adapter | `saves/<运行名>/` |
| 合并模型 | `models/<merged_prefix>-<阶段>-<运行名>/` |

本地中间数据（不进挂载盘包树，成功后可删）：

```
/data/work/<包名>/<运行名>/
├── data_<cutoff>/           # 按 cutoff 过滤后的数据副本
└── output/                  # 训练 checkpoint；成功发布 adapter 后清理
```

## settings.yaml 路径约定（每包各写一份）

新包复制模板后，把路径指到**本包目录**，不要再写全局的 `/cloud/cloud-ssd1/saves/...` 或 `/cloud/cloud-ssd1/models`：

```yaml
project:
  name: GoldThread                       # 或本批数据的显示名
  merged_prefix: Qwen3.8-27B-goldthread  # 合并目录名 = <merged_prefix>-<阶段>-<运行名>

base_model: /model/ModelScope/Qwen/Qwen3.8-27B
work_dir: /data/work/<包名>                            # 本地，按包隔离
saves_dir: /cloud/cloud-ssd1/packages/<包名>/saves     # 本包 adapter
merged_dir: /cloud/cloud-ssd1/packages/<包名>/models    # 本包合并模型
```

`autotrain.py` 只读本目录的 `settings.yaml`，因此**在哪个包目录里启动，就只写进哪个包**。

## 上传新数据后怎么建包

1. 从轻量模板复制（不要直接改正在用的包，也不要从含模型的包 `cp -a`，以免多拷约 100GB）：

   ```bash
   PKG=goldthread-$(date +%Y%m%d)          # 或自定义别名
   cp -a /cloud/cloud-ssd1/goldthread-train \
         /cloud/cloud-ssd1/packages/$PKG
   rm -rf /cloud/cloud-ssd1/packages/$PKG/runs/*
   rm -f  /cloud/cloud-ssd1/packages/$PKG/issues/pending.jsonl
   mkdir -p /cloud/cloud-ssd1/packages/$PKG/{saves,models}
   mkdir -p /data/work/$PKG
   ```

2. 放入上传文件：覆盖或替换 `packages/$PKG/data/` 下的 jsonl，更新 `dataset_info.json` 与 `MANIFEST.json`。
3. 改 `packages/$PKG/settings.yaml`：`project` 段 + 上面的 `work_dir` / `saves_dir` / `merged_dir`（路径里的包名一并改掉）。
4. 校验：`cd packages/$PKG && python tools/check_data.py`。
5. 开训：`cd packages/$PKG && nohup python autotrain.py sft > runs/sft.out 2>&1 &`

模板里的 `issues/known_issues.json`、`configs/`、`tools/` 建议保留；`runs/` 与 `pending.jsonl` 必须清空，避免串台。

## 怎么找某次产物

1. 按上传日期或数据别名进 `packages/<包名>/`。
2. 看合并模型：`packages/<包名>/models/`（部署通常取最新的 `*-dpo-*`）。
3. 看 adapter：`packages/<包名>/saves/<运行名>/`。
4. 看这次怎么训的：`packages/<包名>/runs/history.jsonl` 或 `python autotrain.py status`（在该包目录下执行）。

同一包内多次重训（例如改了方案再跑）会生成多个 `<运行名>` 子目录，彼此并列，互不覆盖；以 `history.jsonl` 里 `status: ok` 且带 `merged` 路径的为准。

## 与旧布局的关系

历史上成果曾散落在挂载盘根下的 `saves/`、`models/`，以及模板目录的 `runs/`。现已废弃这些全局路径；**一律只用** `packages/<包名>/` 包内的 `data/` / `runs/` / `saves/` / `models/`。本机已迁移示例：`packages/goldthread-20261106/`。

## 磁盘与清理（按包）

- 每个合并模型约 52GB。部署只要 DPO 合并模型时，可删同包内的 SFT 合并模型腾空间（SFT adapter 仍在 `saves/`，可再合并）。
- 删包 = 删掉整棵 `packages/<包名>/`（及可选的 `/data/work/<包名>/`）。删除前必须先问用户。
- 开训前的余量检查仍由 `settings.yaml` 的 `min_free_gb_*` 控制；路径已按包隔离后，清理提示也会落在本包目录附近。
