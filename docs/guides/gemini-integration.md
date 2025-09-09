# Gemini LLM Integration Guide

## 概覽

本指南說明如何在 echo-roots 中使用 Google Gemini LLM 進行分類法萃取。我們已經完全整合了官方的 `google.genai` SDK，提供了強大的多提供者LLM支持架構。

## 快速開始

### 1. 安装依賴

```bash
# 安装包含 LLM 支持的 echo-roots
pip install echo-roots[llm]

# 或者安装單獨的 Gemini SDK
pip install google-genai>=0.3.0
```

### 2. 設定 API 金鑰

```bash
# 設定環境變數
export GOOGLE_API_KEY="your_gemini_api_key_here"

# 或在代碼中設定
api_key = "your_gemini_api_key_here"
```

### 3. 基本使用

```python
from echo_roots.pipelines.llm_factory import create_llm_client
from echo_roots.pipelines.extraction import ExtractionPipeline
from echo_roots.models.core import IngestionItem

# 創建 Gemini 客戶端
client = create_llm_client("gemini")

# 創建萃取管道
pipeline = ExtractionPipeline(
    domains_path="domains",
    llm_client=client
)

# 創建待處理項目
item = IngestionItem(
    item_id="test_001",
    title="Apple iPhone 15 Pro Max",
    description="最新旗艦智慧型手機，搭載 A17 Pro 晶片",
    language="zh",
    source="test"
)

# 執行萃取
result = await pipeline.extract(item, domain="electronics")

# 檢視結果
for attr in result.attributes:
    print(f"{attr.name}: {attr.value}")
```

## 進階配置

### 自定義模型配置

```python
from echo_roots.pipelines.gemini_client import GeminiClient

# 使用特定模型
client = GeminiClient(
    api_key="your_key",
    model_name="gemini-1.5-pro",  # 使用更強大的模型
    project_id="your-project-id"
)

# 自定義生成參數
response = await client.complete(
    prompt="萃取品牌資訊...",
    temperature=0.1,    # 降低隨機性
    max_tokens=1000,    # 限制輸出長度
    top_p=0.8,         # 調整詞彙選擇策略
    top_k=40           # 限制候選詞彙數量
)
```

### 配置檔案方式

創建 `config/llm_config.yaml`:

```yaml
llm:
  provider: "gemini"

gemini:
  api_key: "${GOOGLE_API_KEY}"
  model_name: "gemini-1.5-flash"
  project_id: "your-project-id"
  
  # 生成配置
  generation_config:
    temperature: 0.1
    top_p: 0.8
    top_k: 40
    max_output_tokens: 2000

# 其他提供者配置
openai:
  api_key: "${OPENAI_API_KEY}"
  model_name: "gpt-4"

anthropic:
  api_key: "${ANTHROPIC_API_KEY}"
  model_name: "claude-3-sonnet-20240229"
```

使用配置檔案：

```python
from echo_roots.pipelines.llm_factory import LLMClientFactory

# 從配置檔案創建客戶端
client = LLMClientFactory.create_from_config_file(
    "config/llm_config.yaml", 
    provider="gemini"
)
```

### 多提供者支持

```python
from echo_roots.pipelines.llm_factory import get_available_providers

# 檢查可用的提供者
providers = get_available_providers()
for provider, (available, status) in providers.items():
    if available:
        print(f"✅ {provider}: {status}")
    else:
        print(f"❌ {provider}: {status}")

# 動態選擇提供者
def get_best_available_client():
    if get_available_providers()["gemini"][0]:
        return create_llm_client("gemini")
    elif get_available_providers()["openai"][0]:
        return create_llm_client("openai")
    else:
        return create_llm_client("mock")  # 測試用
```

## 批次處理

```python
# 批次萃取多個項目
items = [
    IngestionItem(item_id="1", title="MacBook Pro M3", ...),
    IngestionItem(item_id="2", title="iPad Air", ...),
    IngestionItem(item_id="3", title="AirPods Pro", ...),
]

# 並行處理
results = await pipeline.extract(items, domain="electronics")

# 處理結果
for result in results:
    print(f"項目 {result.item_id}:")
    for attr in result.attributes:
        print(f"  {attr.name}: {attr.value}")
```

## 錯誤處理

```python
from echo_roots.pipelines.extraction import ExtractionError

try:
    result = await pipeline.extract(item, domain="electronics")
except ExtractionError as e:
    print(f"萃取失敗: {e.message}")
    if e.item_id:
        print(f"項目 ID: {e.item_id}")
    if e.cause:
        print(f"原因: {e.cause}")
```

## 效能調優

### 1. 模型選擇
- `gemini-1.5-flash`: 快速處理，適合批量作業
- `gemini-1.5-pro`: 高品質輸出，適合複雜萃取
- `gemini-1.0-pro`: 平衡效能和成本

### 2. 批次大小調整

```python
from echo_roots.pipelines.extraction import ExtractorConfig

config = ExtractorConfig(
    batch_size=20,          # 增加批次大小
    timeout_seconds=60,     # 延長超時時間
    retry_attempts=2,       # 減少重試次數
    temperature=0.05        # 降低隨機性
)

extractor = LLMExtractor(domain_pack, client, config)
```

### 3. 快取和重用

```python
# 重用客戶端實例
client = create_llm_client("gemini")

# 多個萃取器共用同一個客戶端
extractor1 = LLMExtractor(domain_pack1, client)
extractor2 = LLMExtractor(domain_pack2, client)
```

## 測試和驗證

### 執行整合測試

```bash
# 測試架構（無需 API 金鑰）
python test_llm_architecture.py

# 測試真實 API（需要 GOOGLE_API_KEY）
python test_gemini_integration.py
```

### 單元測試

```python
import pytest
from echo_roots.pipelines.llm_factory import LLMClientFactory

@pytest.mark.asyncio
async def test_gemini_client():
    # 使用測試金鑰或跳過測試
    if not os.getenv("GOOGLE_API_KEY"):
        pytest.skip("需要 GOOGLE_API_KEY 環境變數")
    
    client = LLMClientFactory.create_client("gemini")
    response = await client.complete("測試提示")
    assert len(response) > 0
```

## 故障排除

### 常見問題

1. **ImportError: No module named 'google.genai'**
   ```bash
   pip install google-genai>=0.3.0
   ```

2. **ValueError: Google API key is required**
   ```bash
   export GOOGLE_API_KEY="your_key_here"
   ```

3. **ExtractionError: Gemini API call failed**
   - 檢查網路連接
   - 確認 API 金鑰有效
   - 檢查配額限制

4. **JSON parsing errors**
   - 模型回應格式問題，調整 temperature
   - 使用更詳細的提示詞

### 除錯模式

```python
import logging

# 啟用詳細日誌
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("echo_roots.pipelines.gemini_client")
logger.setLevel(logging.DEBUG)

# 檢查客戶端狀態
from echo_roots.pipelines.llm_factory import LLMClientFactory

available, status = LLMClientFactory.check_provider_availability("gemini")
print(f"Gemini 可用性: {available}, 狀態: {status}")
```

## API 參考

### GeminiClient

```python
class GeminiClient:
    def __init__(self, api_key=None, model_name="gemini-1.5-flash", project_id=None)
    async def complete(self, prompt: str, **kwargs) -> str
    async def generate_batch(self, prompts: List[str], **kwargs) -> List[str]
```

### LLMClientFactory

```python
class LLMClientFactory:
    @classmethod
    def create_client(cls, provider: str, config: dict = None, **kwargs)
    @classmethod  
    def create_from_config_file(cls, config_path: str, provider: str = None)
    @classmethod
    def create_from_environment(cls, provider: str = None)
    @classmethod
    def list_providers(cls) -> Dict[str, str]
    @classmethod
    def check_provider_availability(cls, provider: str) -> tuple[bool, str]
```

## 最佳實踐

1. **安全性**
   - 使用環境變數儲存 API 金鑰
   - 不要在代碼中硬編碼金鑰
   - 定期輪換 API 金鑰

2. **效能**
   - 使用批次處理提高效率
   - 根據需求選擇合適的模型
   - 實施適當的重試和超時機制

3. **成本控制**
   - 監控 API 使用量
   - 使用較小的模型進行開發測試
   - 實施快取機制避免重複請求

4. **品質保證**
   - 驗證萃取結果的格式
   - 實施信心分數評估
   - 建立評測基準數據

## 相關資源

- [Google AI Studio](https://makersuite.google.com/)
- [Gemini API 文檔](https://ai.google.dev/docs)
- [echo-roots 文檔](./ARCHITECTURE.md)
- [域配置指南](./guides/domain-configuration.md)
