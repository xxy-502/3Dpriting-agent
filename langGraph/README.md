# 离线 3D 打印缺陷诊断 Agent

该目录包含一个基于 LangGraph 的确定性工作流。大模型不负责猜测应调用哪个
视觉工具；媒体类型、节点路由、失败处理和是否生成结果视频都由程序确定。
Qwen2.5 仅在检测与本地检索完成后生成最终中文分析。

> Python 包名为 `langGraph`。

## 工作流

```text
用户请求
   |
   v
prepare_request（解析媒体路径与打印参数）
   |----------------------|----------------------|
   v                      v                      v
image_detection       video_detection       用户已给检测结果
(YOLO + SAM)              |                      |
                          v                      |
                 video_yolo_detection           |
                          |                      |
              有缺陷且要求结果视频？             |
                     | yes        | no           |
                     v            |              |
              video_sam_rendering |              |
                     |------------|--------------|
                                  v
                   retrieve_knowledge（BGE + BM25 + RRF）
                                  |
                                  v
                       generate_answer（本地 Qwen）
                                  |
                                  v
                  数值一致性校验；发现错误时按需重写
```

视频中的 `defect_count` 表示经过时序确认的缺陷事件数，而不是所有帧中检测框的
简单累加。完整逐帧结果保存在 `yolo_defect_events.json`，LangGraph 状态只保存可
JSON 序列化的摘要和文件路径，不保存帧、掩码、模型或 CUDA 张量。

## 本地资源

- Qwen：`models/Qwen`（支持模型文件直接存放或 `snapshots/master` 布局）
- YOLO：自动查找 `models/yolo` 中的 `.pt` 文件
- SAM2：自动查找 `models/sam` 中的 `.pt` 文件
- BGE：自动查找 `models/bge` 中的本地 Transformers 模型
- Qdrant：`langGraph/Qdrant_collection`
- BM25 词表：随 `Qdrant_collection` 一起加载

所有模型均以 `local_files_only=True` 或本地文件路径加载，运行过程不访问网络。
本地 Qdrant 存储不适合被多个进程同时写入或打开；一个服务进程中应复用同一组
`AgentServices`。

## 命令行运行

从仓库根目录运行。Windows PowerShell 示例：

```powershell
conda run --no-capture-output -n pytorch python -m langGraph.agent `
  --prompt "请检测图片并分析。材料PLA，喷嘴0.4mm，喷嘴温度215C。" `
  --media-path "D:\data\part.jpg" `
  --llm-device cuda --vision-device cuda --rag-device cpu
```

Linux 示例：

```bash
conda run --no-capture-output -n pytorch python -m langGraph.agent \
  --prompt '请检测视频并分析。材料PLA，喷嘴0.4mm。' \
  --media-path /data/short_print.mp4 \
  --render-result-video \
  --llm-device cuda --vision-device cuda --rag-device cpu
```

未传 `--prompt` 时进入交互模式。`--json` 输出完整图状态；默认只打印最终回答和
结果媒体路径。视频诊断默认只运行 YOLO，只有传入 `--render-result-video`、检测
成功且确有缺陷时才加载 SAM2 并输出掩码视频。

如果用户已经有检测结果，可不传媒体：

```powershell
conda run --no-capture-output -n pytorch python -m langGraph.agent `
  --prompt "检测结果发现2处拉丝，材料PLA，喷嘴温度220C，回抽距离0.8mm，请分析"
```

## Python 接入

```python
from langGraph import build_defect_agent

graph = build_defect_agent(
    llm_device="cuda",
    vision_device="cuda",
    rag_device="cpu",
    video_coarse_stride=7,
)

result = graph.invoke(
    {
        "user_query": "检测该视频并结合知识库分析，材料PLA，喷嘴0.4mm",
        "media_path": "/data/short_print.mp4",
        "render_result_video": True,
        "output_dir": "/data/results/job-001",
    }
)

print(result["final_answer"])
print(result.get("detection_result"))
print(result.get("sources"))
print(result.get("result_video_path"))
```

入口和主要模块：

- `agent_graph.py`：总图、默认模型路径与常驻服务构建
- `agent_nodes.py`：请求、视觉、RAG、回答和一致性校验逻辑
- `request_parser.py`：Windows/Linux 路径与常用打印参数解析
- `rag_retriever.py`：离线 BGE + BM25 + Qdrant RRF 检索
- `local_qwen.py`：本地 Qwen2.5 推理
- `image_defect/`：图片 YOLO-OBB + SAM2 工具
- `video/`：两阶段视频子图与工具

## 测试

单元测试不加载模型：

```powershell
conda run --no-capture-output -n pytorch python -m unittest discover -s langGraph\tests -v
```

真实视频冒烟测试请先截取数秒或几十帧，不要直接使用完整视频：

```powershell
conda run --no-capture-output -n pytorch python -m langGraph.video.video_graph `
  "langGraph\runs\smoke\input_frames_220_270.mp4" `
  --output-dir "langGraph\runs\validation\video_subgraph" `
  --render-result-video
```

若 SAM2 提示本地扩展 `_C` 不可用，它会跳过可选的孔洞后处理；当前工程中的主体
分割和视频传播仍可继续运行。
