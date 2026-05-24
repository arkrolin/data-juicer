# DJ 当前支持的资源相关环境变量

## 1. 说明

这份文档整理的是 **当前 Data-Juicer 已支持、且和资源下载 / 资源缓存 / 模型加载直接相关** 的环境变量。

重点覆盖：

- cache 目录
- 新增的资源治理变量
- HuggingFace 相关变量
- 运行时自动装包开关
- OpenAI / DashScope 兼容接口常用变量

不覆盖：

- Ray / CUDA / 业务功能类零散环境变量
- 单个工具内部临时使用的 env

## 2. 现有 cache 目录变量

这些变量是 DJ 之前就支持的，主要负责“资源存在哪里”。

定义位置：
- [cache_utils.py](/Users/dludora/Code/data-juicer/data_juicer/utils/cache_utils.py)

### `DATA_JUICER_CACHE_HOME`
- 含义：DJ 缓存根目录
- 默认值：`~/.cache/data_juicer`
- 用途：作为 models / assets 默认缓存根

### `DATA_JUICER_ASSETS_CACHE`
- 含义：assets 缓存目录
- 默认值：`$DATA_JUICER_CACHE_HOME/assets`
- 用途：
  - 静态词表缓存
  - 部分 repo 工作目录
  - 部分中间产物目录

### `DATA_JUICER_MODELS_CACHE`
- 含义：models 缓存目录
- 默认值：`$DATA_JUICER_CACHE_HOME/models`
- 用途：模型文件缓存

### `DATA_JUICER_EXTERNAL_MODELS_HOME`
- 含义：额外的外部模型目录
- 默认值：`None`
- 用途：给 `check_model()` / `check_model_home()` 提供补充本地查找路径
- 支持多个路径，使用 `os.pathsep` 分隔

## 3. 新增的资源治理变量

这些变量是本轮新增的，主要负责“从哪里拉、是否允许联网、是否允许回退公网”。

定义位置：
- [resource_policy_utils.py](/Users/dludora/Code/data-juicer/data_juicer/utils/resource_policy_utils.py)

### `DJ_RESOURCE_OFFLINE_MODE`
- 含义：是否开启严格离线模式
- 默认值：`false`
- 支持值：`1/0`、`true/false`、`yes/no`、`on/off`
- 行为：
  - `true` 时禁止公网 fallback
  - `true` 时禁止运行时自动装包
  - `true` 时禁止 NLTK 在线下载

### `DJ_RESOURCE_ALLOW_PUBLIC_FALLBACK`
- 含义：本地或镜像未命中时，是否允许回退到当前默认公网源
- 默认值：`true`
- 支持值：`1/0`、`true/false`、`yes/no`、`on/off`
- 说明：
  - 仅在 `DJ_RESOURCE_OFFLINE_MODE=false` 时生效
  - 若 `offline_mode=true`，它会被强制视为 `false`

### `DJ_RESOURCE_LOCAL_CACHE_ROOTS`
- 含义：额外的本地共享资源根目录
- 默认值：空
- 格式：多个路径使用 `os.pathsep` 分隔
- 当前用途：
  - `model` 查找补充路径
  - `asset` 查找补充路径

### `DJ_RESOURCE_BASE_URL`
- 含义：统一的资源镜像根前缀
- 默认值：空
- 当前用途：
  - 作为 `DJ_MODEL_BASE_URL` 和 `DJ_ASSET_BASE_URL` 的统一上层入口
- 推导规则：
  - 若未显式设置 `DJ_MODEL_BASE_URL`，则自动推导为 `{resource_base_url.rstrip('/')}/models`
  - 若未显式设置 `DJ_ASSET_BASE_URL`，则自动推导为 `{resource_base_url.rstrip('/')}`
- 适用场景：
  - 原来挂在同一个 `data_juicer/` 前缀下的资源整体迁移到新的内网 OSS

### `DJ_MODEL_BASE_URL`
- 含义：模型文件镜像根地址
- 默认值：空
- 当前用途：
  - 覆盖 `check_model()` 这条线的默认下载源
- 当前默认源：
  - `MODEL_LINKS`
  - `BACKUP_MODEL_LINKS`

### `DJ_ASSET_BASE_URL`
- 含义：静态词表镜像根地址
- 默认值：空
- 当前用途：
  - 覆盖 `flagged_words.json`
  - 覆盖 `stopwords.json`
- 当前拼接规则：
  - `flagged_words` -> `{base_url.rstrip('/')}/flagged_words.json`
  - `stopwords` -> `{base_url.rstrip('/')}/stopwords.json`

## 4. HuggingFace 相关变量

这些变量用于控制 `from_pretrained(...)` 这条加载链。

### `DJ_HF_ENDPOINT`
- 含义：HuggingFace 镜像地址
- 默认值：空
- 当前行为：
  - 若设置，则在运行时写入 `HF_ENDPOINT`

### `DJ_HF_HOME`
- 含义：HuggingFace cache 根目录
- 默认值：空
- 当前行为：
  - 若设置，则在运行时写入 `HF_HOME`

### `DJ_HF_LOCAL_FILES_ONLY`
- 含义：是否强制 HF 只从本地 cache / 本地目录加载
- 默认值：空
- 支持值：`1/0`、`true/false`、`yes/no`、`on/off`
- 当前行为：
  - 若为 `true`，则设置：
    - `HF_HUB_OFFLINE=1`
    - `TRANSFORMERS_OFFLINE=1`
- 额外说明：
  - 若 `DJ_RESOURCE_OFFLINE_MODE=true`，当前也会设置这两个离线标记

## 5. NLTK 相关变量

### `DJ_NLTK_DATA_DIR`
- 含义：NLTK 数据目录
- 默认值：空
- 当前行为：
  - 若设置，则插入 `nltk.data.path`

### `DJ_NLTK_ALLOW_DOWNLOAD`
- 含义：是否允许 `nltk.download(...)`
- 默认值：`true`
- 支持值：`1/0`、`true/false`、`yes/no`、`on/off`
- 当前行为：
  - 为 `false` 时，`ensure_nltk_resource()` 不再联网下载
  - 若 `DJ_RESOURCE_OFFLINE_MODE=true`，也会禁止下载

## 6. 运行时自动装包变量

### `DJ_PACKAGE_AUTO_INSTALL`
- 含义：缺包时是否允许自动安装
- 默认值：`true`
- 支持值：`1/0`、`true/false`、`yes/no`、`on/off`
- 当前行为：
  - 为 `false` 时，`LazyLoader.check_packages(...)` 发现缺包直接报错
  - 若 `DJ_RESOURCE_OFFLINE_MODE=true`，也会被强制视为 `false`

说明：
- 当前 DJ **不再额外封装 pip index 环境变量**
- 如果用户要控制 pip 源，直接使用 pip / uv 自己支持的环境变量即可

## 7. OpenAI / DashScope 兼容接口相关变量

这部分不是本轮新增，但 DJ 当前已经支持，并且和“模型从哪里访问”有关。

主要使用位置：
- [model_utils.py](/Users/dludora/Code/data-juicer/data_juicer/utils/model_utils.py)

### `OPENAI_BASE_URL`
- 用途：OpenAI 兼容接口 base URL
- 典型场景：接 DashScope 兼容接口或自建 OpenAI-compatible 服务

### `OPENAI_API_URL`
- 用途：兼容 `OPENAI_BASE_URL` 的另一种写法

### `OPENAI_API_KEY`
- 用途：OpenAI 兼容接口认证

### `DASHSCOPE_BASE_URL`
- 用途：DashScope base URL

### `DASHSCOPE_API_KEY`
- 用途：DashScope 认证

### `DASHSCOPE_DEFAULT_MODEL`
- 用途：DashScope 默认模型名重映射

### `OPENAI_DEFAULT_MODEL`
- 用途：OpenAI 兼容接口默认模型名重映射

### `SK`
- 用途：部分兼容调用里也会作为 API key 兜底读取

## 8. 当前新变量主要接管了哪些默认前缀

### 文件型模型
- 默认前缀：
```text
https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/data_juicer/models/
```
- 可通过：
  - `DJ_RESOURCE_BASE_URL`
  - `DJ_MODEL_BASE_URL`
  - `DJ_RESOURCE_LOCAL_CACHE_ROOTS`
  - `DATA_JUICER_EXTERNAL_MODELS_HOME`
  - `DJ_RESOURCE_OFFLINE_MODE`
  - `DJ_RESOURCE_ALLOW_PUBLIC_FALLBACK`
控制

### 词表资源
- 默认地址：
```text
https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/data_juicer/flagged_words.json
https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/data_juicer/stopwords.json
```
- 可通过：
  - `DJ_RESOURCE_BASE_URL`
  - `DJ_ASSET_BASE_URL`
  - `DJ_RESOURCE_LOCAL_CACHE_ROOTS`
  - `DJ_RESOURCE_OFFLINE_MODE`
  - `DJ_RESOURCE_ALLOW_PUBLIC_FALLBACK`
控制

### HuggingFace 仓式加载
- 默认来源：HuggingFace Hub
- 可通过：
  - `DJ_HF_ENDPOINT`
  - `DJ_HF_HOME`
  - `DJ_HF_LOCAL_FILES_ONLY`
  - `DJ_RESOURCE_OFFLINE_MODE`
控制

## 9. 当前不在这轮环境变量里统一接管的内容

这些资源目前还没有统一的专属 env 入口：

- GitHub repo clone 地址
- repo mirror 地址
- 直接写死的部分 HuggingFace 文件下载 URL
- pip index / extra index 地址

这些是下一阶段可以继续补的内容。

## 10. 推荐用法

### 10.1 纯离线运行

```bash
export DJ_RESOURCE_OFFLINE_MODE=true
export DJ_NLTK_ALLOW_DOWNLOAD=false
export DJ_PACKAGE_AUTO_INSTALL=false
```

### 10.2 使用统一内网 OSS 前缀

```bash
export DJ_RESOURCE_BASE_URL=https://your-internal-oss.example.com/data_juicer
```

### 10.3 单独覆写模型或词表镜像

```bash
export DJ_RESOURCE_BASE_URL=https://your-internal-oss.example.com/data_juicer
export DJ_MODEL_BASE_URL=https://your-internal-oss.example.com/custom-models
export DJ_ASSET_BASE_URL=https://your-internal-oss.example.com/custom-assets
```

说明：
- `DJ_MODEL_BASE_URL` / `DJ_ASSET_BASE_URL` 优先级高于 `DJ_RESOURCE_BASE_URL`

### 10.4 使用内网 HuggingFace Mirror

```bash
export DJ_HF_ENDPOINT=https://your-hf-mirror.example.com
export DJ_HF_HOME=/data/hf-cache
```

### 10.5 使用共享本地资源目录

```bash
export DJ_RESOURCE_LOCAL_CACHE_ROOTS=/mnt/dj-share/models:/mnt/dj-share/assets
```
