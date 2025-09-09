#!/usr/bin/env python3
"""
實際使用您的 LLM API 運行 Echo-Roots 提取的腳本
Real LLM API integration script for Echo-Roots
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import aiohttp

# 添加 src 到路徑
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from echo_roots.models.domain import DomainPack
    from echo_roots.models.core import IngestionItem, AttributeExtraction, SemanticTerm, ExtractionResult, ExtractionMetadata
    from echo_roots.storage.duckdb_backend import DuckDBStorageManager
    from echo_roots.domain.adapter import DomainAdapter
except ImportError as e:
    print(f"❌ 無法導入 Echo-Roots 模組: {e}")
    print("請確保您在正確的目錄中運行此腳本")
    sys.exit(1)


class GeminiLLMClient:
    """Google Gemini API 客戶端"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash", **kwargs):
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = 'https://generativelanguage.googleapis.com/v1beta'
        self.session = None
    
    async def complete(self, prompt: str, **kwargs) -> str:
        """發送請求到 Gemini API"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        # Gemini API 使用 URL 參數傳遞 API key
        url = f'{self.base_url}/models/{self.model_name}:generateContent?key={self.api_key}'
        
        headers = {
            'Content-Type': 'application/json'
        }
        
        # Gemini API 的請求格式
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": f"You are a product data extraction expert. Return valid JSON only.\n\n{prompt}"
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": kwargs.get('temperature', 0.3),
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": kwargs.get('max_tokens', 2000),
                "stopSequences": []
            }
        }
        
        try:
            async with self.session.post(
                url,
                headers=headers,
                json=payload,
                timeout=30
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # 解析 Gemini 的回應格式
                    if 'candidates' in data and len(data['candidates']) > 0:
                        candidate = data['candidates'][0]
                        if 'content' in candidate and 'parts' in candidate['content']:
                            parts = candidate['content']['parts']
                            if len(parts) > 0 and 'text' in parts[0]:
                                return parts[0]['text'].strip()
                    
                    raise Exception("Invalid response format from Gemini API")
                else:
                    error_text = await response.text()
                    raise Exception(f"Gemini API Error {response.status}: {error_text}")
                    
        except Exception as e:
            raise Exception(f"Gemini API call failed: {str(e)}")
    
    async def close(self):
        if self.session:
            await self.session.close()


class OpenAILLMClient:
    """OpenAI API 客戶端"""
    
    def __init__(self, api_key: str, model_name: str = "gpt-4", **kwargs):
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = kwargs.get('base_url', 'https://api.openai.com/v1')
        self.temperature = kwargs.get('temperature', 0.3)
        self.max_tokens = kwargs.get('max_tokens', 2000)
        self.session = None
    
    async def complete(self, prompt: str, **kwargs) -> str:
        """發送請求到 OpenAI API"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': self.model_name,
            'messages': [
                {
                    'role': 'system', 
                    'content': 'You are a product data extraction expert. Return valid JSON only.'
                },
                {
                    'role': 'user', 
                    'content': prompt
                }
            ],
            'temperature': kwargs.get('temperature', self.temperature),
            'max_tokens': kwargs.get('max_tokens', self.max_tokens)
        }
        
        try:
            async with self.session.post(
                f'{self.base_url}/chat/completions',
                headers=headers,
                json=payload,
                timeout=30
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['choices'][0]['message']['content'].strip()
                else:
                    error_text = await response.text()
                    raise Exception(f"OpenAI API Error {response.status}: {error_text}")
                    
        except Exception as e:
            raise Exception(f"OpenAI API call failed: {str(e)}")
    
    async def close(self):
        """關閉會話"""
        if self.session:
            await self.session.close()


class RealLLMExtractor:
    """使用真實 LLM API 的提取器"""
    
    def __init__(self, domain_pack: DomainPack, llm_client):
        self.domain_pack = domain_pack
        self.llm_client = llm_client
        
    def build_extraction_prompt(self, item: IngestionItem) -> str:
        """構建提取提示"""
        # 使用域包中的提示模板
        extraction_prompt = self.domain_pack.llm_prompts.get('extraction', '')
        category_prompt = self.domain_pack.llm_prompts.get('category_classification', '')
        
        # 構建完整提示
        prompt = f"""
{extraction_prompt}

Product Information:
Title: {item.title}
Description: {item.description}
Content: {item.raw_content}

請以以下 JSON 格式回應:
{{
    "attributes": [
        {{"name": "attribute_name", "value": "attribute_value", "confidence": 0.95}}
    ],
    "terms": [
        {{"term": "technical_term", "confidence": 0.90}}
    ],
    "category": "predicted_category"
}}

{category_prompt}
"""
        return prompt
    
    async def extract_single(self, item: IngestionItem) -> ExtractionResult:
        """對單一項目進行提取"""
        import time
        start_time = time.time()
        
        try:
            # 構建提示
            prompt = self.build_extraction_prompt(item)
            
            # 調用 LLM
            response = await self.llm_client.complete(prompt)
            
            # 解析回應
            try:
                result_data = json.loads(response)
            except json.JSONDecodeError:
                # 如果不是有效 JSON，嘗試提取 JSON 部分
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    result_data = json.loads(json_match.group())
                else:
                    raise Exception("無法從 LLM 回應中解析 JSON")
            
            # 創建屬性提取結果
            attributes = []
            for attr_data in result_data.get('attributes', []):
                attributes.append(AttributeExtraction(
                    name=attr_data['name'],
                    value=attr_data['value'],
                    confidence=attr_data.get('confidence', 0.8)
                ))
            
            # 創建術語提取結果
            terms = []
            for term_data in result_data.get('terms', []):
                terms.append(SemanticTerm(
                    term=term_data['term'],
                    confidence=term_data.get('confidence', 0.8)
                ))
            
            # 創建元數據
            processing_time = int((time.time() - start_time) * 1000)
            metadata = ExtractionMetadata(
                model=self.llm_client.model_name,
                run_id=f"run_{int(time.time())}",
                extracted_at=datetime.now(),
                processing_time_ms=processing_time
            )
            
            return ExtractionResult(
                item_id=item.id,
                attributes=attributes,
                terms=terms,
                predicted_category=result_data.get('category', ''),
                metadata=metadata
            )
            
        except Exception as e:
            # 創建失敗結果
            processing_time = int((time.time() - start_time) * 1000)
            metadata = ExtractionMetadata(
                model=self.llm_client.model_name,
                run_id=f"run_{int(time.time())}",
                extracted_at=datetime.now(),
                processing_time_ms=processing_time
            )
            
            return ExtractionResult(
                item_id=item.id,
                attributes=[],
                terms=[],
                predicted_category='',
                metadata=metadata
            )


def load_config() -> Dict[str, Any]:
    """載入配置"""
    provider = os.getenv('LLM_PROVIDER', 'gemini')  # 預設使用 Gemini
    
    if provider == 'gemini':
        config = {
            'provider': 'gemini',
            'api_key': os.getenv('GOOGLE_API_KEY', ''),
            'model_name': os.getenv('GEMINI_MODEL', 'gemini-1.5-flash'),
            'temperature': 0.3,
            'max_tokens': 2000
        }
        
        if not config['api_key']:
            print("❌ 請設定 GOOGLE_API_KEY 環境變數")
            print("您可以:")
            print("1. 前往 https://aistudio.google.com/app/apikey 獲取 API 金鑰")
            print("2. 執行: export GOOGLE_API_KEY='your-gemini-api-key'")
            print("3. 或在 .env 文件中設定 GOOGLE_API_KEY")
            return None
            
    elif provider == 'openai':
        config = {
            'provider': 'openai',
            'api_key': os.getenv('OPENAI_API_KEY', ''),
            'model_name': os.getenv('OPENAI_MODEL', 'gpt-4'),
            'temperature': 0.3,
            'max_tokens': int(os.getenv('OPENAI_MAX_TOKENS', '2000'))
        }
        
        if not config['api_key']:
            print("❌ 請設定 OPENAI_API_KEY 環境變數")
            return None
    else:
        print(f"❌ 不支援的 LLM 提供商: {provider}")
        return None
    
    return config


async def main():
    """主函數"""
    print("🤖 Echo-Roots 真實 LLM API 整合測試")
    print("=" * 50)
    
    # 載入配置
    config = load_config()
    if not config:
        return
    
    # 創建 LLM 客戶端
    print(f"🔗 連接到 {config['provider']} {config['model_name']}...")
    
    if config['provider'] == 'gemini':
        llm_client = GeminiLLMClient(
            api_key=config['api_key'],
            model_name=config['model_name']
        )
    elif config['provider'] == 'openai':
        llm_client = OpenAILLMClient(
            api_key=config['api_key'],
            model_name=config['model_name'],
            temperature=config['temperature'],
            max_tokens=config['max_tokens']
        )
    else:
        print(f"❌ 不支援的 LLM 提供商: {config['provider']}")
        return
    
    try:
        # 載入領域包
        print("📦 載入電商領域包...")
        domain_path = Path("domains/ecommerce/domain.yaml")
        if not domain_path.exists():
            print(f"❌ 找不到領域包文件: {domain_path}")
            return
        
        domain_adapter = DomainAdapter.from_file(domain_path)
        domain_pack = domain_adapter.domain_pack
        
        # 初始化存儲
        print("💾 連接到資料庫...")
        storage_config = {
            "duckdb": {
                "database_path": "./test_echo_roots.db"
            }
        }
        
        storage_manager = DuckDBStorageManager(storage_config)
        await storage_manager.initialize()
        
        # 創建提取器
        extractor = RealLLMExtractor(domain_pack, llm_client)
        
        # 獲取測試項目
        print("📋 獲取測試資料...")
        items = await storage_manager.ingestion.list_items(limit=3)
        
        if not items:
            print("❌ 沒有找到測試資料。請先運行 test_data_processing.py")
            return
        
        print(f"✅ 找到 {len(items)} 個測試項目")
        
        # 運行提取
        print(f"\n🔄 開始使用 {config['model_name']} 進行提取...")
        
        total_attributes = 0
        total_terms = 0
        total_time = 0
        
        for i, item in enumerate(items, 1):
            print(f"\n📝 處理項目 {i}: {item.title}")
            
            try:
                result = await extractor.extract_single(item)
                
                if result.metadata.error_message:
                    print(f"❌ 提取失敗: {result.metadata.error_message}")
                    continue
                
                # 顯示結果
                print(f"✅ 提取了 {len(result.attributes)} 個屬性:")
                for attr in result.attributes:
                    print(f"  - {attr.name}: {attr.value} (信心度: {attr.confidence:.2f})")
                    total_attributes += 1
                
                print(f"✅ 提取了 {len(result.terms)} 個術語:")
                for term in result.terms[:3]:  # 只顯示前 3 個
                    print(f"  - {term.term} (信心度: {term.confidence:.2f})")
                    total_terms += 1
                
                if result.predicted_category:
                    print(f"🏷️ 預測類別: {result.predicted_category}")
                
                print(f"⏱️ 處理時間: {result.metadata.processing_time_ms}ms")
                total_time += result.metadata.processing_time_ms
                
                # 保存結果到資料庫（可選）
                # await storage_manager.extraction.save_result(result)
                
            except Exception as e:
                print(f"❌ 處理項目時發生錯誤: {str(e)}")
        
        # 顯示統計
        print(f"\n📊 提取統計:")
        print(f"  - 總屬性數: {total_attributes}")
        print(f"  - 總術語數: {total_terms}")
        print(f"  - 平均處理時間: {total_time / len(items):.0f}ms")
        print(f"  - 使用模型: {config['model_name']}")
        
    except Exception as e:
        print(f"❌ 發生錯誤: {str(e)}")
        
    finally:
        # 清理資源
        print("\n🧹 清理資源...")
        await llm_client.close()
        await storage_manager.close()
        
    print("\n🎉 LLM 整合測試完成！")


if __name__ == "__main__":
    # 檢查依賴
    try:
        import aiohttp
    except ImportError:
        print("❌ 需要安裝 aiohttp: pip install aiohttp")
        sys.exit(1)
    
    asyncio.run(main())
