# 安装

OMG 面向 Python 3.10 和支持 CUDA 的 Linux 计算机，用于训练、ONNX 导出、
TensorRT 推理和 HoloMotion 跟踪。macOS 可用于轻量级代码审查和文档工作，
但 GPU 执行应在 Linux 上进行。

## Python 环境

请使用仓库内的本地虚拟环境。

```bash
cd /path/to/OMG
curl -LsSf https://astral.sh/uv/install.sh | sh  # 如果已经安装 uv，请跳过
uv venv --python 3.10 .venv
source .venv/bin/activate
```

先安装 PyTorch，再安装 OMG。

```bash
uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install -e ".[all]"
```

也可以通过 Makefile 完成相同的设置：

```bash
make venv
source .venv/bin/activate
make install
```

中国大陆网络环境请使用：

```bash
make install-cn
```

如需精简安装，请按任务选择额外依赖：

```bash
uv pip install -e ".[train]"
uv pip install -e ".[data]"
uv pip install -e ".[render]"
uv pip install -e ".[tracking]"
uv pip install -e ".[export]"
uv pip install -e ".[realtime]"
uv pip install -e ".[benchmark]"
```

常用运行时环境变量：

```bash
export PYTHONPATH=src
export TOKENIZERS_PARALLELISM=false
```

## 数据与模型根目录

发布配置的默认路径为：

```text
data/OMG-Data
models/
```

使用外部磁盘或共享存储时，可通过以下变量覆盖：

```bash
export OMG_DATA_ROOT=/path/to/OMG-Data
export OMG_MATERIALIZED_ROOT=/path/to/OMG-Data/materialized
export OMG_MODELS_ROOT=/path/to/OMG-models
```

文本条件训练和生成需要 Hugging Face `t5-base` 文本编码器。
默认配置从以下位置加载：

```text
${OMG_MODELS_ROOT}/t5-base-local
```

离线或集群运行时，请将 `t5-base` 的本地副本放在该位置；也可以使用其他本地路径
或 Hugging Face 模型 ID 覆盖 `model.text_encoder.model_name`。

如果文本编码器已缓存，并且计算机不应访问网络：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

## HoloMotion 依赖

离线跟踪器和实时部署需要：

- HoloMotion 动作跟踪 ONNX 模型。
- G1 MuJoCo XML 或默认的 HoloMotion G1 场景。
- `onnxruntime-gpu` 和 TensorRT 运行时库。
- 用于离线仿真和渲染的 MuJoCo。

请从 [HoloMotion 官方仓库](https://github.com/HorizonRobotics/HoloMotion)或
[HoloMotion Hugging Face 制品](https://huggingface.co/HorizonRobotics/HoloMotion_models)
下载 HoloMotion G1 动作跟踪 ONNX 模型。OMG 不会再分发 HoloMotion 权重。
推荐的本地路径为：

```text
models/holomotion/motion_tracking/model.onnx
```

运行跟踪或流水线模式时，请通过 `--holomotion-onnx` 显式传入此路径。

实体机器人部署时，建议优先使用 HoloMotion 速度跟踪模型，并将其放在：

```text
models/holomotion/velocity_tracking/model.onnx
```

各任务的运行命令请参阅[生成](generation.md)、[跟踪](tracking.md)和
[G1 实时部署](realtime_g1.md)。
