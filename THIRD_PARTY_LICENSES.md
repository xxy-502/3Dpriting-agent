# Third-Party Licenses and Notices

本项目是个人学术研究和非商业分享项目。下列第三方软件、源码和模型权重仍分别受其原始许可证约束。本文件是便于使用者理解和定位原始条款的说明，不构成法律意见，也不会替代各项目随附或官方发布的完整许可证文本。

如果许可证摘要与官方许可证存在差异，以官方许可证原文为准。

## 1. Segment Anything 2 / SAM 2.1 Hiera Base+

- 组件：`sam2/` 中的 SAM 2 源码、Hydra 配置，以及下载到 `models/sam/sam2.1_hiera_base_plus.pt` 的官方权重。
- 作者/权利人：Meta Platforms, Inc. 及贡献者。
- 许可证：Apache License 2.0。
- 官方项目：https://github.com/facebookresearch/sam2
- 官方许可证：https://github.com/facebookresearch/sam2/blob/main/LICENSE
- 官方权重：https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt

Apache 2.0 通常允许在保留许可证、版权、专利、商标和 NOTICE 等相关声明并标明修改的前提下使用、修改和分发。商标权不随该许可证授予。分发 `sam2/` 源码或官方权重时，请同时保留 Meta 原始文件中的版权头并附带 Apache 2.0 完整许可证。

## 2. Ultralytics YOLO26 与自训练权重

- 组件：运行时依赖 `ultralytics`；本仓库中的 `models/yolo/yolo_best.pt` 是项目作者使用 Ultralytics YOLO26 OBB 框架训练的自定义权重，不是由下载脚本获取的官方预训练权重。
- Ultralytics 作者/权利人：Ultralytics 及贡献者。
- 开源许可证：GNU Affero General Public License v3.0（AGPL-3.0）。
- 官方项目：https://github.com/ultralytics/ultralytics
- 官方许可证：https://github.com/ultralytics/ultralytics/blob/main/LICENSE
- 官方许可说明：https://www.ultralytics.com/license

AGPL-3.0 是强 copyleft 许可证，并包含网络服务场景下向用户提供相应源代码的义务。Ultralytics 同时提供企业许可选项。非商业、研究或免费分享本身不会自动免除 AGPL-3.0 的条件；若将本项目用于闭源、商业产品或公开网络服务，应先评估 AGPL-3.0 的适用要求，必要时向 Ultralytics 获取适当许可。

自训练权重还可能受到训练数据集许可、标注来源、个人信息和其他权利的影响。权重提供者应确保自己有权分发训练数据产生的模型，并单独说明训练数据许可。本文件不对训练数据或自训练权重作额外授权。

## 3. Qwen2.5-3B-Instruct

- 组件：由 `download_models.py` 下载到 `models/Qwen/Qwen2.5-3B-Instruct/` 的模型配置、分词器和权重。
- 作者/权利人：Alibaba Cloud / Qwen Team。
- 许可证：Qwen Research License Agreement，发布日期 2024-09-19。
- 官方模型：https://huggingface.co/Qwen/Qwen2.5-3B-Instruct
- 官方许可证原文：https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE

该特定 3B 模型的官方模型页标记为 `qwen-research`，不是 Apache 2.0。许可证将“非商业”定义为仅用于研究或评估，并仅授予非商业目的的使用、复制、修改和分发权；商业使用需要另行向许可方申请授权。

如果再分发 Qwen 材料或其衍生内容，官方条款还要求向接收者提供该协议、标明修改，并保留指定的 Qwen 归属声明。若使用 Qwen 材料或输出创建、训练、微调或改进并对外提供另一个 AI 模型，还应检查官方条款中的 “Built with Qwen” 或 “Improved using Qwen” 标示要求。下载或使用即表示使用者应自行阅读并接受官方完整协议。

本项目的个人学术研究和非商业分享定位与该许可证的预期非商业范围一致，但使用者仍须独立确保自己的具体使用方式符合全部条款。

## 4. BAAI BGE Small EN v1.5

- 组件：由 `download_models.py` 下载到 `models/bge/bge-small-en-v1.5/` 的模型配置、分词器和权重。
- 作者/权利人：Beijing Academy of Artificial Intelligence（BAAI）及贡献者。
- 许可证：MIT License。
- 官方模型：https://huggingface.co/BAAI/bge-small-en-v1.5
- 官方项目：https://github.com/FlagOpen/FlagEmbedding
- MIT 许可证：https://opensource.org/license/mit

MIT License 通常允许使用、复制、修改、合并、发布、分发、再许可和销售软件副本，但须在软件的重要部分中保留版权声明和许可声明。软件按“现状”提供，不附带保证。

## 5. Python 依赖

本项目还依赖 PyTorch、Torchvision、LangGraph、LangChain Core、Transformers、Hugging Face Hub、Qdrant Client、Ultralytics、OpenCV、NumPy、Pillow、Hydra、OmegaConf、iopath、tqdm、Pydantic 和 Safetensors 等软件包。它们由包管理器安装，不作为本仓库源码的一部分分发，并分别受各自许可证约束。使用者可以通过以下命令查看已安装包的元数据：

```bash
python -m pip show torch torchvision langgraph langchain-core transformers huggingface-hub qdrant-client ultralytics opencv-python numpy Pillow hydra-core omegaconf iopath tqdm pydantic safetensors
```

## 6. 项目用途声明与免责声明

本项目维护者仅将此项目用于个人学术研究、教学演示和非商业技术分享。该用途声明不是一个能够覆盖第三方材料的统一许可证，也不会缩减或扩大任何第三方权利。

任何计划进行商业化、闭源部署、公开网络服务、模型再分发或大规模数据处理的使用者，都应自行复核当时有效的官方条款并在必要时寻求专业法律意见。
