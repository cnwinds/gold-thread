# 传输说明

只传 **1 个文件**到训练机：`goldthread-train-bundle.tar.gz`（4.1 MB）

```bash
# 本机（或你的电脑）上传
scp dist/goldthread-train-bundle.tar.gz root@<训练机IP>:/data/

# 训练机上解包
mkdir -p /data/goldthread
tar -xzf /data/goldthread-train-bundle.tar.gz -C /data/goldthread/
ls /data/goldthread/03_train | head
```

校验（可选，确认传输没损坏）：

```bash
sha256sum /data/goldthread-train-bundle.tar.gz
# 期望：fef1ac7af30e23702e6c8b8f62d5c553bccee50b02344205c6d6750162dd0268
```

然后从 `/data/goldthread/03_train/START_HERE.md` 开始。
