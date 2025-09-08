# Echo-Roots LLM API 整合指南

## 📋 整合概覽

您現在有*### 🎯 推薦執行順序 (使用 Gemini)

### 步驟 1: 設定 Gemini API
```bash
# 1. 設定 Gemini API 金鑰
export GOOGLE_API_KEY='your-gemini-api-key'
export LLM_PROVIDER='gemini'

# 2. 測試 Gemini 基本功能
python3 test_gemini_api.py
```

### 步驟 2: 快速驗證 (通用)
```bash
# 測試基本 LLM 功能 (支援多種提供商)
python3 test_llm_api.py
```

### 步驟 3: 完整測試
如果前面步驟成功，執行完整整合：
```bash
python3 run_real_llm.py
``` API 到 Echo-Roots 系統中：

### 1. 🚀 快速測試 - `test_llm_api.py`
**用途**: 快速驗證您的 LLM API 是否工作正常
```bash
# 設定 API 金鑰
export OPENAI_API_KEY='your-actual-api-key'

# 執行測試
python3 test_llm_api.py
```

**特點**:
- 獨立測試，不依賴 Echo-Roots 複雜模組
- 測試產品屬性提取和分類功能
- 快速驗證 API 連接和回應格式

### 2. 🔧 完整整合 - `run_real_llm.py`
**用途**: 將您的 LLM API 整合到完整的 Echo-Roots 管道中
```bash
# 設定 API 金鑰
export OPENAI_API_KEY='your-actual-api-key'

# 執行完整整合測試
python3 run_real_llm.py
```

**特點**:
- 使用真實的 Echo-Roots 資料和存儲
- 完整的領域包整合
- 處理您準備的 50 個產品資料

### 3. ⚙️ 自訂整合 - `llm_api_integration.py`
**用途**: 參考文檔，整合您特定的 LLM 提供商
- 支援 OpenAI, Anthropic, Azure OpenAI, Google Vertex AI
- 自訂 API 整合範例
- 生產環境配置指南

---

## 🎯 推薦執行順序

### 步驟 1: 快速驗證
```bash
# 1. 設定 API 金鑰
export OPENAI_API_KEY='your-actual-api-key'

# 2. 測試基本功能
python3 test_llm_api.py
```

### 步驟 2: 完整測試
如果步驟 1 成功，執行完整整合：
```bash
python3 run_real_llm.py
```

### 步驟 3: 自訂配置
根據需要參考 `llm_api_integration.py` 進行自訂整合。

---

## 📁 相關檔案

| 檔案 | 用途 | 狀態 |
|------|------|------|
| `config/llm_config.yaml` | LLM 配置文件 | ✅ 已建立 (支援 Gemini) |
| `.env.example` | 環境變數範例 | ✅ 已更新 (包含 Gemini) |
| `test_gemini_api.py` | Gemini 專用測試 | ✅ 新建立 |
| `test_llm_api.py` | 快速 API 測試 | ✅ 已建立 |
| `run_real_llm.py` | 完整整合測試 | ✅ 已更新 (支援 Gemini) |
| `llm_api_integration.py` | 整合參考文檔 | ✅ 已建立 |

---

## 🔑 API 金鑰設定

### Google Gemini API (推薦)
```bash
# 設定 Gemini API 金鑰
export GOOGLE_API_KEY='your-gemini-api-key'
export GEMINI_MODEL='gemini-1.5-flash'

# 或設定為預設 LLM 提供商
export LLM_PROVIDER='gemini'
```

**如何獲取 Gemini API 金鑰:**
1. 前往 [Google AI Studio](https://aistudio.google.com/app/apikey)
2. 登入您的 Google 帳號
3. 點擊 "Create API Key"
4. 複製生成的 API 金鑰

### OpenAI API
```bash
export OPENAI_API_KEY='your-openai-api-key'
export LLM_PROVIDER='openai'
```

### 其他提供商
```bash
export ANTHROPIC_API_KEY='your-anthropic-key'  # 如果使用 Anthropic
```

### 方法 2: .env 檔案
```bash
# 複製範例檔案
cp .env.example .env

# 編輯 .env 檔案並填入您的 API 金鑰
nano .env
```

---

## 📊 測試預期結果

### 成功的測試輸出範例:
```
🤖 測試 LLM API 產品屬性提取
==================================================

📱 測試產品 1: iPhone 15 Pro 256GB 黑色
✅ 提取成功 (耗時: 1205ms)
   品牌: Apple
   型號: iPhone 15 Pro
   顏色: 黑色
   容量: 256GB
   螢幕: 6.1吋
   等級: premium
   類別: 智慧型手機
   特色: A17 Pro晶片, 三鏡頭相機系統, 5G支援

🎯 使用模型: gpt-4
```

---

## 🛠️ 故障排除

### 常見問題:

1. **`ModuleNotFoundError: No module named 'aiohttp'`**
   ```bash
   pip install aiohttp
   ```

2. **`API Error 401: Unauthorized`**
   - 檢查 API 金鑰是否正確
   - 確認 API 金鑰有足夠的權限

3. **`API Error 429: Rate limit exceeded`**
   - 降低請求頻率
   - 檢查 API 使用配額

4. **JSON 解析錯誤**
   - LLM 回應格式不正確
   - 調整提示模板以確保 JSON 輸出

---

## 🚀 下一步

當基本測試成功後，您可以：

1. **調整提示模板**: 修改 `domains/ecommerce/domain.yaml` 中的 LLM 提示
2. **擴展屬性提取**: 添加更多產品屬性到提取邏輯
3. **優化性能**: 調整 `temperature`, `max_tokens` 等參數
4. **生產部署**: 設定正式的 API 金鑰和錯誤處理

---

## 💡 技術支援

如果遇到問題，請檢查：
1. API 金鑰是否有效
2. 網路連接是否正常  
3. LLM 服務是否可用
4. 請求格式是否符合 API 規範

測試成功後，您就可以在 Echo-Roots 中使用真實的 LLM API 進行產品資料提取和分類了！
