# 资源相关环境变量

本文档描述 Data-Juicer 当前支持的、与资源下载、资源缓存和模型加载相关的环境变量。

## 概述

Data-Juicer 中与资源访问相关的环境变量可以分为四类：

- 资源镜像地址：控制模型、词表等默认资源从哪里下载
- 本地缓存目录：控制资源在本地存放在哪里，以及额外从哪些本地目录查找
- 运行策略开关：控制是否允许回退公网、是否允许在线下载、是否允许自动安装依赖
- 模型服务兼容接口：控制 HuggingFace、OpenAI 兼容接口和 DashScope 的访问地址

如果你的目标是把 DJ 默认使用的 OSS 资源切换到内网地址，通常优先设置下面这一项：

```bash
export DJ_RESOURCE_BASE_URL=https://your-internal-oss.example.com/data_juicer
```

它会影响：

- 模型默认前缀：`{DJ_RESOURCE_BASE_URL}/models`
- 词表默认前缀：`{DJ_RESOURCE_BASE_URL}`

## 资源镜像地址

### `DJ_RESOURCE_BASE_URL`

统一的资源镜像根前缀。

当 DJ 默认资源整体迁移到同一个内网 OSS 前缀时，优先设置这个变量即可。当前会自动推导：

- `DJ_MODEL_BASE_URL = {DJ_RESOURCE_BASE_URL.rstrip('/')}/models`
- `DJ_ASSET_BASE_URL = {DJ_RESOURCE_BASE_URL.rstrip('/')}`

例如：

```bash
export DJ_RESOURCE_BASE_URL=https://your-internal-oss.example.com/data_juicer
```

### `DJ_MODEL_BASE_URL`

模型文件镜像根地址。

用于覆盖 DJ 默认模型下载前缀。若同时设置了 `DJ_RESOURCE_BASE_URL` 和 `DJ_MODEL_BASE_URL`，以 `DJ_MODEL_BASE_URL` 为准。

当前默认模型前缀为：

```text
https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/data_juicer/models/
```

例如：

```bash
export DJ_MODEL_BASE_URL=https://your-internal-oss.example.com/custom-models
```

### `DJ_ASSET_BASE_URL`

静态词表镜像根地址。

用于覆盖 `flagged_words.json` 和 `stopwords.json` 的默认下载地址。若同时设置了 `DJ_RESOURCE_BASE_URL` 和 `DJ_ASSET_BASE_URL`，以 `DJ_ASSET_BASE_URL` 为准。

当前默认词表地址为：

```text
https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/data_juicer/flagged_words.json
https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/data_juicer/stopwords.json
```

例如：

```bash
export DJ_ASSET_BASE_URL=https://your-internal-oss.example.com/custom-assets
```

## 运行策略开关

### `DJ_RESOURCE_OFFLINE_MODE`

是否开启严格离线模式。默认值为 `false`。

支持值：

- `1/0`
- `true/false`
- `yes/no`
- `on/off`

开启后：

- 不再回退到默认公网源
- 不再允许 NLTK 在线下载
- 不再允许运行时自动安装依赖

### `DJ_RESOURCE_ALLOW_PUBLIC_FALLBACK`

本地或镜像未命中时，是否允许回退到默认公网源。默认值为 `true`。

支持值：

- `1/0`
- `true/false`
- `yes/no`
- `on/off`

说明：

- 仅在 `DJ_RESOURCE_OFFLINE_MODE=false` 时生效
- 若 `DJ_RESOURCE_OFFLINE_MODE=true`，此项会被视为 `false`

### `DJ_NLTK_ALLOW_DOWNLOAD`

是否允许 `nltk.download(...)`。默认值为 `true`。

支持值：

- `1/0`
- `true/false`
- `yes/no`
- `on/off`

说明：

- 若 `DJ_RESOURCE_OFFLINE_MODE=true`，此项会被视为 `false`

### `DJ_PACKAGE_AUTO_INSTALL`

缺包时是否允许自动安装。默认值为 `true`。

支持值：

- `1/0`
- `true/false`
- `yes/no`
- `on/off`

说明：

- 若 `DJ_RESOURCE_OFFLINE_MODE=true`，此项会被视为 `false`

## 本地缓存目录

这些变量主要控制资源存放位置和本地查找路径。

定义位置：

- [cache_utils.py](/Users/dludora/Code/data-juicer/data_juicer/utils/cache_utils.py)

### `DATA_JUICER_CACHE_HOME`

DJ 缓存根目录。默认值为 `~/.cache/data_juicer`。

### `DATA_JUICER_ASSETS_CACHE`

Assets 缓存目录。默认值为 `$DATA_JUICER_CACHE_HOME/assets`。

通常用于：

- 静态词表缓存
- 部分 repo 工作目录
- 部分中间产物

### `DATA_JUICER_MODELS_CACHE`

Models 缓存目录。默认值为 `$DATA_JUICER_CACHE_HOME/models`。

### `DATA_JUICER_EXTERNAL_MODELS_HOME`

额外的外部模型目录。默认值为 `None`。

用于给 `check_model()` / `check_model_home()` 提供补充本地查找路径。支持多个路径，使用 `os.pathsep` 分隔。

### `DJ_RESOURCE_LOCAL_CACHE_ROOTS`

额外的本地共享资源根目录。默认值为空。

支持多个路径，使用 `os.pathsep` 分隔。当前会参与：

- 模型本地查找
- 词表本地查找

例如：

```bash
export DJ_RESOURCE_LOCAL_CACHE_ROOTS=/mnt/dj-share/models:/mnt/dj-share/assets
```

## HuggingFace 相关变量

这些变量用于控制 `from_pretrained(...)` 这条加载链。

### `DJ_HF_ENDPOINT`

HuggingFace 镜像地址。

若设置，则运行时写入 `HF_ENDPOINT`。

### `DJ_HF_HOME`

HuggingFace 缓存根目录。

若设置，则运行时写入 `HF_HOME`。

### `DJ_HF_LOCAL_FILES_ONLY`

是否强制 HuggingFace 只从本地缓存或本地目录加载。

支持值：

- `1/0`
- `true/false`
- `yes/no`
- `on/off`

若为 `true`，运行时会设置：

- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`

当 `DJ_RESOURCE_OFFLINE_MODE=true` 时，当前也会设置这两个离线标记。

## NLTK 相关变量

### `DJ_NLTK_DATA_DIR`

NLTK 数据目录。

若设置，则插入 `nltk.data.path`，优先从该目录查找资源。

## OpenAI / DashScope 兼容接口

这部分不是文件资源下载开关，但与模型访问地址直接相关。

主要使用位置：

- [model_utils.py](/Users/dludora/Code/data-juicer/data_juicer/utils/model_utils.py)

### `OPENAI_BASE_URL`

OpenAI 兼容接口的 base URL。

### `OPENAI_API_URL`

兼容 `OPENAI_BASE_URL` 的另一种写法。

### `OPENAI_API_KEY`

OpenAI 兼容接口认证。

### `DASHSCOPE_BASE_URL`

DashScope base URL。

### `DASHSCOPE_API_KEY`

DashScope 认证。

### `DASHSCOPE_DEFAULT_MODEL`

DashScope 默认模型名重映射。

### `OPENAI_DEFAULT_MODEL`

OpenAI 兼容接口默认模型名重映射。

### `SK`

部分兼容调用里也会作为 API key 兜底读取。

## 常用配置示例

### 使用统一内网 OSS 前缀

```bash
export DJ_RESOURCE_BASE_URL=https://your-internal-oss.example.com/data_juicer
```

### 单独覆写模型或词表镜像

```bash
export DJ_RESOURCE_BASE_URL=https://your-internal-oss.example.com/data_juicer
export DJ_MODEL_BASE_URL=https://your-internal-oss.example.com/custom-models
export DJ_ASSET_BASE_URL=https://your-internal-oss.example.com/custom-assets
```

### 纯离线运行

```bash
export DJ_RESOURCE_OFFLINE_MODE=true
export DJ_NLTK_ALLOW_DOWNLOAD=false
export DJ_PACKAGE_AUTO_INSTALL=false
```

### 使用内网 HuggingFace Mirror

```bash
export DJ_HF_ENDPOINT=https://your-hf-mirror.example.com
export DJ_HF_HOME=/data/hf-cache
```
