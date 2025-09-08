#!/usr/bin/env python3
"""
簡化版 LLM API 測試腳本
用於快速驗證您的 LLM API 整合
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
import aiohttp

# 設定 API 金鑰 (請替換為您的實際金鑰)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'your-api-key-here')
MODEL_NAME = os.getenv('OPENAI_MODEL', 'gpt-4')


class SimpleLLMClient:
    """簡化的 LLM 客戶端，用於快速測試"""
    
    def __init__(self, api_key: str, model_name: str = "gpt-4"):
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = 'https://api.openai.com/v1'
        self.session = None
    
    async def complete(self, prompt: str) -> str:
        """發送請求到 LLM API"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': self.model_name,
            'messages': [
                {'role': 'system', 'content': 'You are a helpful assistant. Return valid JSON only.'},
                {'role': 'user', 'content': prompt}
            ],
            'temperature': 0.3,
            'max_tokens': 1000
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
                    raise Exception(f"API Error {response.status}: {error_text}")
                    
        except Exception as e:
            raise Exception(f"LLM API call failed: {str(e)}")
    
    async def close(self):
        if self.session:
            await self.session.close()


async def test_product_extraction():
    """測試產品屬性提取"""
    
    print("🤖 測試 LLM API 產品屬性提取")
    print("=" * 50)
    
    # 檢查 API 金鑰
    if OPENAI_API_KEY == 'your-api-key-here':
        print("❌ 請設定您的 OPENAI_API_KEY 環境變數")
        print("執行: export OPENAI_API_KEY='your-actual-api-key'")
        return
    
    # 創建 LLM 客戶端
    llm_client = SimpleLLMClient(OPENAI_API_KEY, MODEL_NAME)
    
    # 測試產品資料
    test_products = [
        {
            "title": "iPhone 15 Pro 256GB 黑色",
            "description": "蘋果最新旗艦手機，6.1吋超級視網膜XDR顯示器，A17 Pro晶片，三鏡頭相機系統，支援5G"
        },
        {
            "title": "Samsung Galaxy S24 Ultra",
            "description": "三星頂級智慧型手機，6.8吋AMOLED螢幕，Snapdragon 8 Gen 3處理器，S Pen手寫筆"
        },
        {
            "title": "MacBook Air M2 13吋 8GB/256GB 太空灰",
            "description": "蘋果筆記型電腦，M2晶片，13.6吋Liquid Retina顯示器，全天候電池續航力"
        }
    ]
    
    try:
        for i, product in enumerate(test_products, 1):
            print(f"\n📱 測試產品 {i}: {product['title']}")
            
            # 構建提取提示
            prompt = f"""
請從以下產品資訊中提取結構化屬性：

產品標題: {product['title']}
產品描述: {product['description']}

請以 JSON 格式回應，包含以下欄位：
{{
    "brand": "品牌名稱",
    "model": "型號",
    "color": "顏色",
    "storage": "儲存容量",
    "screen_size": "螢幕尺寸",
    "price_tier": "價格等級 (budget/mid-range/premium)",
    "category": "產品類別",
    "key_features": ["主要特色1", "主要特色2", "主要特色3"]
}}

只回傳有效的 JSON，不要包含其他文字。
"""
            
            start_time = time.time()
            
            # 調用 LLM
            response = await llm_client.complete(prompt)
            
            processing_time = int((time.time() - start_time) * 1000)
            
            # 解析回應
            try:
                result = json.loads(response)
                
                print(f"✅ 提取成功 (耗時: {processing_time}ms)")
                print(f"   品牌: {result.get('brand', 'N/A')}")
                print(f"   型號: {result.get('model', 'N/A')}")
                print(f"   顏色: {result.get('color', 'N/A')}")
                print(f"   容量: {result.get('storage', 'N/A')}")
                print(f"   螢幕: {result.get('screen_size', 'N/A')}")
                print(f"   等級: {result.get('price_tier', 'N/A')}")
                print(f"   類別: {result.get('category', 'N/A')}")
                
                features = result.get('key_features', [])
                if features:
                    print(f"   特色: {', '.join(features[:3])}")
                
            except json.JSONDecodeError:
                print(f"❌ JSON 解析失敗")
                print(f"原始回應: {response[:200]}...")
    
        print(f"\n🎯 使用模型: {MODEL_NAME}")
        
    except Exception as e:
        print(f"❌ 測試失敗: {str(e)}")
        
    finally:
        await llm_client.close()


async def test_category_classification():
    """測試產品分類"""
    
    print("\n🏷️ 測試產品分類")
    print("-" * 30)
    
    if OPENAI_API_KEY == 'your-api-key-here':
        return
    
    llm_client = SimpleLLMClient(OPENAI_API_KEY, MODEL_NAME)
    
    test_items = [
        "Nike Air Max 270 男款運動鞋",
        "Sony WH-1000XM5 無線降噪耳機",
        "Dyson V15 Detect 無線吸塵器",
        "美的電磁爐IH智能觸控面板"
    ]
    
    # 可用類別 (簡化版)
    categories = [
        "服裝配件", "鞋類", "電子產品", "家電用品", 
        "運動用品", "美容保養", "家居用品", "汽車用品"
    ]
    
    try:
        for item in test_items:
            prompt = f"""
請將以下產品分類到最合適的類別中：

產品: {item}

可用類別: {', '.join(categories)}

請回傳 JSON 格式:
{{
    "predicted_category": "最合適的類別",
    "confidence": 0.95
}}
"""
            
            response = await llm_client.complete(prompt)
            
            try:
                result = json.loads(response)
                category = result.get('predicted_category', 'N/A')
                confidence = result.get('confidence', 0.0)
                
                print(f"📦 {item}")
                print(f"   → {category} (信心度: {confidence:.2f})")
                
            except json.JSONDecodeError:
                print(f"❌ 分類失敗: {item}")
    
    except Exception as e:
        print(f"❌ 分類測試失敗: {str(e)}")
        
    finally:
        await llm_client.close()


async def main():
    """主測試函數"""
    
    print("🚀 Echo-Roots LLM API 整合測試")
    print("=" * 50)
    
    # 檢查依賴
    try:
        import aiohttp
    except ImportError:
        print("❌ 需要安裝 aiohttp: pip install aiohttp")
        return
    
    # 執行測試
    await test_product_extraction()
    await test_category_classification()
    
    print("\n✅ 測試完成！")
    print("\n下一步：")
    print("1. 如果測試成功，您可以將這個 LLM 客戶端整合到 Echo-Roots 中")
    print("2. 使用 run_real_llm.py 進行完整的 Echo-Roots 整合測試")
    print("3. 根據需要調整提示模板和參數")


if __name__ == "__main__":
    asyncio.run(main())
