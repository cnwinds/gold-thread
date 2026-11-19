# 配置治理第一期 · 验收记录

本期只解决配置是否可信、是否可解释。不迁移项目规则、不改变黄金/SFT 准入政策、不修改训练超参数。编码与回归测试由 `bailian/deepseek-v4.1-flash-1017` 执行；以下结论来自独立差异核对、临时工作区请求和本地测试，而非编码模型自报。

| 用户可验证的要求 | 验收证据 |
|---|---|
| 拼错键、类型不对、比例越界或跨字段非法时，CLI 和 Web 都说明错误；不能把保存成功留到下一次运行才发现问题。 | 临时配置分别经 `gt inspect config --json`、校验接口、原文 PUT；逐例核对字段、失败结果和原文件字节。结构化请求含一项有效和一项无效的修改时，两项均不落盘。显式传入不存在的文件失败，不影响未指定文件时使用内建配置。 |
| 用户能分辨文件写的值、默认值、适用阶段，以及离线主链的强制覆盖。 | CLI 与 Web 返回同一份叶字段说明。独立对照抓到过 `llm.max_tokens` 被误当成凭据隐藏；最终对照必须包含这个字段，并验证合法文件可保存。|
| 指定工作区的辅助请求不得读写另一个工作区。 | 用合成模型清单和假 HTTP 返回检查模型路由、预算、超时、缓存文件和命中；切换 provider/端点不沿用旧回复。检查 `pipeline.run --config … --force` 在成功和失败时分别写指定目录，记录实际 force 值。 |
| 保留已有配置与训练基线。 | 加载当前 `config/pipeline.yaml`，与文件/内建默认值合并结果对照；最终核对仓库 YAML 的 SHA-256。历史项目特例、训练模板、真实数据不属于本期编辑对象。 |

**审核退回项**：清除 CLI/Web 的临时宽松解析后备；修正动态键的来源判定；只接受可序列化的有效配置，避免保存后才在哈希/响应时报错；约束原始标签及子段白名单三要素；隔离 HTTP 回归的写入路径。一次早期 HTTP 测试误写了仓库配置，已还原，交付前用测试前的 SHA-256 逐字节核对，不能以“Git 看起来没改”代替这项检查。

**仍待下一期**：Stage 2 增量指纹与 `min_gold_steps` 的入集契约、历史项目根任务替换迁移、学生独立留出与效果归因。本期没有宣称这三个问题已修复，也未做 GPU 调参或学生复训。

## 独立验证

- `python -m pytest tests/test_config_contract.py tests/test_config_http_contract.py tests/test_config_inspect.py tests/test_config_isolation.py tests/test_config_read_guard.py -q`：**121 passed**。这五个文件均为本次新增，分别覆盖共享配置契约、HTTP/前端交互、只读说明、工作区隔离、配置读失败保护。
- `python -m pytest tests/ -q`：**517 passed**，仅 1 个既有的 Starlette/AnyIO 弃用警告。
- `python -m py_compile pipeline/common/config.py pipeline/common/config_schema.py pipeline/common/config_info.py pipeline/common/llm.py pipeline/run.py pipeline/turnclassify.py pipeline/judge.py goldthread/cli.py server/app.py`：通过。提取 `web/index.html` 的内联脚本运行 `node --check`：通过。
- 额外沙箱验收：独立 CLI 子进程和 FastAPI TestClient 对无效配置给出一致结果；合法修改生效；无效写入不动旧字节；未认证编辑不允许改变受保护路径；无需改变该路径即可修复其他语义错误；稀疏配置中的可选键、带点的项目名和负 seed 均按真实消费语义处理。
- 仓库配置：`python gt.py inspect config --json` 输出 93 个叶值，其中 77 个来自文件。`config/pipeline.yaml` 的 SHA-256 为 `48d3e92c9eda493b1e89530a337abd538bcb6aea5f4827d84bde48c668ddd67e`，与执行前一致；未提交或推送。
