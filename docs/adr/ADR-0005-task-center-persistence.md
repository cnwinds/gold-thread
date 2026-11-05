# ADR-0005：任务中心与 Job 内存模型

- **状态**：已接受
- **日期**：2026-11-05

## 背景

长程操作（采集、分析、stage、打包等）通过 `server/app.py` 的 `Job` + 后台线程执行，状态原先仅存于进程内存 `JOBS`。前端用单例 `watchJob` 轮询，并在启动后 **强制跳转「梳理与交付」**，易让用户误以为离开当前工作区；刷新页面后也无法统一查看进行中的任务。

## 决策

1. **Task 是 Job 的展示层 + 磁盘壳**，不另起执行引擎。`task.id === job.id`。
2. **持久化**：创建与结束时双写 `data/00_state/tasks.json`（保留最近 50 条，进行中始终保留）；与 `runs.json` 摘要并存。
3. **API**：`GET /api/tasks`、`GET /api/tasks/{id}`、`POST /api/tasks/{id}/dismiss`；元数据由 `pipeline/task_registry.py` 的 `describe_task` 生成 `title` / `affects` / `links`。
4. **前端 TaskHub** 轮询任务列表（有 running 1.5s / 无 15s）；完成时统一 `onTaskFinished` 按 `affects` 刷新页面；侧栏 **任务中心** 与 **梳理与交付** 分离。
5. **Stale**：服务重启后磁盘上仍为 `running` 且无内存 Job 时标为 `stale`，UI 可 dismiss。

## 后果

- 新增长程操作必须在 `_start` 后 `register_job`，`_run_job` finally 中 `sync_job`。
- 启动异步作业 **不再** 默认 `setTab("runs")`；用户从任务中心查看日志与结果跳转。
- v2 不做：WebSocket、取消任务、重启后 resume。
