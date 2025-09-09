# vLLM Integration Guide

## 概覽

vLLM 是高效能的大型語言模型推理引擎，支援 OpenAI 相容的 HTTP API。本指南說明如何在 echo-roots 中整合 vLLM 進行本地或雲端模型部署。

## 特色

- ✅ **高效能推理**：優化的 CUDA 核心和記憶體管理
- ✅ **OpenAI 相容 API**：無縫整合現有工具鏈
- ✅ **多模型支援**：Llama, Mixtral, Qwen 等開源模型
- ✅ **靈活部署**：本地、雲端、容器化部署
- ✅ **批次處理**：高效率的並行推理

## 快速開始

### 1. 安裝 vLLM

```bash
# 安裝 vLLM (需要 CUDA 支援)
pip install vllm

# 或使用 conda
conda install vllm -c pytorch -c nvidia

# echo-roots 的 OpenAI 相依性
pip install openai
```

### 2. 啟動 vLLM 伺服器

#### 本地部署

```bash
# 啟動 Llama-2-7B 模型
python -m vllm.entrypoints.openai.api_server \\
    --model meta-llama/Llama-2-7b-chat-hf \\
    --port 8000 \\
    --host 0.0.0.0

# 啟動 Qwen 模型 (中文支援)
python -m vllm.entrypoints.openai.api_server \\
    --model Qwen/Qwen-7B-Chat \\
    --port 8000 \\
    --trust-remote-code

# 啟動 Mixtral 模型 (高效能)
python -m vllm.entrypoints.openai.api_server \\
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \\
    --port 8000 \\
    --tensor-parallel-size 2
```

#### Docker 部署

```bash
# 使用官方 Docker 映像
docker run --gpus all \\
    -p 8000:8000 \\
    vllm/vllm-openai:latest \\
    --model meta-llama/Llama-2-7b-chat-hf
```

### 3. 配置環境變數

```bash
# 本地部署
export VLLM_BASE_URL="http://localhost:8000/v1"
export VLLM_MODEL_NAME="meta-llama/Llama-2-7b-chat-hf"
export VLLM_DEPLOYMENT_TYPE="local"

# 雲端部署
export VLLM_BASE_URL="https://your-vllm-endpoint.com/v1"
export VLLM_MODEL_NAME="your-model-name"
export VLLM_API_KEY="your-api-key"
export VLLM_DEPLOYMENT_TYPE="cloud"
```

### 4. 基本使用

```python
from echo_roots.pipelines.llm_factory import create_llm_client
from echo_roots.pipelines.extraction import ExtractionPipeline

# 創建 vLLM 客戶端
client = create_llm_client("vllm")

# 創建萃取管道
pipeline = ExtractionPipeline(llm_client=client)

# 執行萃取
result = await pipeline.extract(item, domain="electronics")
```

## 進階配置

### 配置檔案方式

創建 `config/vllm_config.yaml`:

```yaml
llm:
  provider: "vllm"

vllm:
  # 本地部署配置
  deployment_type: "local"
  host: "localhost"
  port: 8000
  model_name: "meta-llama/Llama-2-7b-chat-hf"
  api_key: "dummy-key"
  
  # 生成參數
  timeout: 60
  max_retries: 3
  
  # 或雲端部署配置
  # deployment_type: "cloud"
  # base_url: "https://your-vllm-endpoint.com/v1"
  # model_name: "your-cloud-model"
  # api_key: "${VLLM_API_KEY}"
  # timeout: 120
  # max_retries: 5
```

### 直接客戶端配置

```python
from echo_roots.pipelines.vllm_client import VLLMLocalClient, VLLMCloudClient

# 本地客戶端
local_client = VLLMLocalClient(
    model_name="meta-llama/Llama-2-7b-chat-hf",
    host="localhost",
    port=8000,
    timeout=60
)

# 雲端客戶端
cloud_client = VLLMCloudClient(
    base_url="https://your-endpoint.com/v1",
    model_name="your-model",
    api_key="your-key",
    timeout=120
)

# 自定義生成參數
response = await client.complete(
    prompt="萃取產品資訊...",
    temperature=0.1,      # 降低隨機性
    max_tokens=2000,      # 最大輸出長度
    top_p=0.9,           # 詞彙選擇策略
    top_k=50,            # vLLM 專用參數
    frequency_penalty=0.0,
    presence_penalty=0.0
)
```

## 多模型支援

### 推薦模型配置

```bash
# 中文優化模型
# Qwen 系列 (阿里巴巴)
python -m vllm.entrypoints.openai.api_server \\
    --model Qwen/Qwen-14B-Chat \\
    --trust-remote-code \\
    --port 8000

# ChatGLM 系列 (清華)
python -m vllm.entrypoints.openai.api_server \\
    --model THUDM/chatglm3-6b \\
    --trust-remote-code \\
    --port 8000

# 多語言通用模型  
# Llama-2 系列 (Meta)
python -m vllm.entrypoints.openai.api_server \\
    --model meta-llama/Llama-2-13b-chat-hf \\
    --port 8000

# Mixtral 系列 (Mistral AI)
python -m vllm.entrypoints.openai.api_server \\
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \\
    --tensor-parallel-size 2 \\
    --port 8000
```

### 動態模型切換

```python
from echo_roots.pipelines.vllm_client import VLLMClient

# 創建多個客戶端實例
clients = {
    "llama2-7b": VLLMLocalClient("meta-llama/Llama-2-7b-chat-hf", port=8000),
    "qwen-14b": VLLMLocalClient("Qwen/Qwen-14B-Chat", port=8001), 
    "mixtral": VLLMLocalClient("mistralai/Mixtral-8x7B-Instruct-v0.1", port=8002)
}

# 根據語言選擇模型
def get_best_model(language: str, complexity: str = "medium"):
    if language in ["zh", "zh-TW", "zh-CN"]:
        return clients["qwen-14b"]
    elif complexity == "high":
        return clients["mixtral"]
    else:
        return clients["llama2-7b"]

# 使用
client = get_best_model(item.language)
result = await pipeline.extract(item, domain="electronics")
```

## 效能調優

### 1. 硬體優化

```bash
# GPU 記憶體優化
python -m vllm.entrypoints.openai.api_server \\
    --model meta-llama/Llama-2-7b-chat-hf \\
    --gpu-memory-utilization 0.8 \\
    --max-num-batched-tokens 8192

# 多 GPU 並行
python -m vllm.entrypoints.openai.api_server \\
    --model meta-llama/Llama-2-13b-chat-hf \\
    --tensor-parallel-size 2 \\
    --pipeline-parallel-size 1

# CPU 卸載 (記憶體不足時)
python -m vllm.entrypoints.openai.api_server \\
    --model meta-llama/Llama-2-7b-chat-hf \\
    --cpu-offload-gb 4
```

### 2. 批次處理優化

```python
from echo_roots.pipelines.extraction import ExtractorConfig

# 優化批次大小和並行度
config = ExtractorConfig(
    batch_size=32,           # 增加批次大小
    timeout_seconds=120,     # 延長超時
    retry_attempts=2,        # 減少重試
    temperature=0.05,        # 降低隨機性
    max_tokens=1500         # 控制輸出長度
)

# 批次萃取
items = [item1, item2, item3, ...]  # 大量項目
results = await pipeline.extract(items, domain="electronics")
```

### 3. 快取策略

```python
import asyncio
from functools import lru_cache

class CachedVLLMClient:
    def __init__(self, base_client):
        self.client = base_client
        self._cache = {}
    
    @lru_cache(maxsize=1000)
    def _get_cache_key(self, prompt: str, **kwargs) -> str:
        return f"{hash(prompt)}:{hash(frozenset(kwargs.items()))}"
    
    async def complete(self, prompt: str, **kwargs) -> str:
        cache_key = self._get_cache_key(prompt, **kwargs)
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        result = await self.client.complete(prompt, **kwargs)
        self._cache[cache_key] = result
        return result

# 使用快取客戶端
base_client = create_llm_client("vllm")
cached_client = CachedVLLMClient(base_client)
```

## 監控和除錯

### 健康檢查

```python
from echo_roots.pipelines.vllm_client import check_vllm_availability

# 檢查 vLLM 可用性
available, status = check_vllm_availability()
print(f"vLLM available: {available}, Status: {status}")

# 客戶端健康檢查
client = create_llm_client("vllm")
healthy = await client.health_check()

# 獲取模型資訊
model_info = await client.get_model_info()
print(f"Model: {model_info}")
```

### 效能監控

```python
import time
import logging

# 啟用詳細日誌
logging.basicConfig(level=logging.DEBUG)

# 效能監控裝飾器
def monitor_performance(func):
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        
        print(f"{func.__name__} took {(end_time - start_time)*1000:.0f}ms")
        return result
    return wrapper

# 使用監控
@monitor_performance
async def extract_with_monitoring(item):
    return await pipeline.extract(item, domain="electronics")
```

### vLLM 伺服器監控

```bash
# 檢查 vLLM 伺服器狀態
curl http://localhost:8000/health

# 獲取模型資訊
curl http://localhost:8000/v1/models

# 監控 GPU 使用率
nvidia-smi -l 1

# 監控記憶體使用
watch -n 1 'free -h'
```

## 故障排除

### 常見問題

1. **CUDA out of memory**
   ```bash
   # 減少 GPU 記憶體使用
   --gpu-memory-utilization 0.6
   
   # 啟用 CPU 卸載
   --cpu-offload-gb 8
   
   # 使用較小的模型
   --model meta-llama/Llama-2-7b-chat-hf  # 而不是 13b
   ```

2. **Connection refused**
   ```bash
   # 檢查伺服器是否運行
   ps aux | grep vllm
   
   # 檢查端口是否開放
   netstat -an | grep 8000
   
   # 重啟 vLLM 伺服器
   pkill -f vllm
   python -m vllm.entrypoints.openai.api_server ...
   ```

3. **Slow inference**
   ```bash
   # 增加批次大小
   --max-num-batched-tokens 16384
   
   # 使用多 GPU
   --tensor-parallel-size 2
   
   # 優化 KV 快取
   --block-size 16
   ```

4. **Model loading errors**
   ```bash
   # 信任遠程代碼 (某些模型需要)
   --trust-remote-code
   
   # 指定修訂版本
   --revision main
   
   # 檢查模型路徑
   huggingface-cli download meta-llama/Llama-2-7b-chat-hf
   ```

### 除錯模式

```python
import os
os.environ["VLLM_DEBUG"] = "1"

# 詳細錯誤資訊
client = create_llm_client("vllm")

try:
    response = await client.complete("test prompt")
except Exception as e:
    print(f"Error details: {e}")
    # 檢查伺服器日誌
```

## 生產部署

### Docker Compose 部署

```yaml
# docker-compose.yml
version: '3.8'
services:
  vllm-server:
    image: vllm/vllm-openai:latest
    ports:
      - "8000:8000"
    environment:
      - CUDA_VISIBLE_DEVICES=0
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    command: >
      --model meta-llama/Llama-2-7b-chat-hf
      --host 0.0.0.0
      --port 8000
      --gpu-memory-utilization 0.8
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
  
  echo-roots:
    build: .
    depends_on:
      - vllm-server
    environment:
      - VLLM_BASE_URL=http://vllm-server:8000/v1
      - VLLM_MODEL_NAME=meta-llama/Llama-2-7b-chat-hf
```

### Kubernetes 部署

```yaml
# vllm-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-server
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm-server
  template:
    metadata:
      labels:
        app: vllm-server
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: 16Gi
          requests:
            nvidia.com/gpu: 1
            memory: 8Gi
        command:
        - python
        - -m
        - vllm.entrypoints.openai.api_server
        args:
        - --model=meta-llama/Llama-2-7b-chat-hf
        - --host=0.0.0.0
        - --port=8000
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-service
spec:
  selector:
    app: vllm-server
  ports:
  - port: 8000
    targetPort: 8000
```

## 測試和驗證

### 執行整合測試

```bash
# 測試 vLLM 整合 (需要運行的 vLLM 伺服器)
python test_vllm_integration.py

# 測試特定配置
export VLLM_BASE_URL="http://localhost:8000/v1"
export VLLM_MODEL_NAME="meta-llama/Llama-2-7b-chat-hf"
python test_vllm_integration.py
```

### 單元測試

```python
import pytest
from echo_roots.pipelines.vllm_client import VLLMLocalClient

@pytest.mark.asyncio
async def test_vllm_local_client():
    # 跳過測試如果沒有本地 vLLM 伺服器
    client = VLLMLocalClient("meta-llama/Llama-2-7b-chat-hf")
    
    if not await client.health_check():
        pytest.skip("vLLM server not available")
    
    response = await client.complete("Hello", max_tokens=10)
    assert len(response) > 0
```

## 最佳實踐

1. **模型選擇**
   - 7B 模型：快速推理，適合批量處理
   - 13B+ 模型：更高品質，適合複雜任務
   - 專用模型：使用針對特定語言優化的模型

2. **資源管理**
   - 監控 GPU 記憶體使用率
   - 實施自動重啟機制
   - 使用負載平衡器處理高並發

3. **安全性**
   - 在生產環境中使用適當的 API 認證
   - 限制網路存取和防火牆規則
   - 定期更新 vLLM 版本

4. **成本優化**
   - 使用較小的模型進行開發測試
   - 實施智能快取策略
   - 考慮使用 CPU 卸載節省記憶體

## 相關資源

- [vLLM 官方文檔](https://docs.vllm.ai/)
- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [支援的模型列表](https://docs.vllm.ai/en/latest/models/supported_models.html)
- [OpenAI API 規格](https://platform.openai.com/docs/api-reference)
- [echo-roots 架構文檔](./ARCHITECTURE.md)
