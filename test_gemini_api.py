#!/usr/bin/env python3
"""
Google Gemini API 測試腳本
專門用於測試 Google Gemini API 整合
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
import aiohttp

# Gemini API 配置
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', 'your-google-api-key-here')
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')
GEMINI_PROJECT_ID = os.getenv('GEMINI_PROJECT_ID', '')


class GeminiLLMClient:
    """Google Gemini API 客戶端"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash", project_id: str = ""):
        self.api_key = api_key
        self.model_name = model_name
        self.project_id = project_id
        self.base_url = 'https://generativelanguage.googleapis.com/v1beta'
        self.session = None
    
    async def complete(self, prompt: str) -> str:
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
                            "text": f"You are a helpful assistant specialized in product data extraction. Please respond with valid JSON only.\n\n{prompt}"
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.3,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 2000,
                "stopSequences": []
            },
            "safetySettings": [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH", 
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
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


async def test_gemini_connection():
    """測試 Gemini API 連線"""
    print("🔗 測試 Gemini API 連線")
    print("-" * 30)
    
    if GOOGLE_API_KEY == 'your-google-api-key-here':
        print("❌ 請設定您的 GOOGLE_API_KEY 環境變數")
        print("執行: export GOOGLE_API_KEY='your-actual-google-api-key'")
        print("\n📝 如何獲取 Gemini API 金鑰:")
        print("1. 前往 https://aistudio.google.com/app/apikey")
        print("2. 登入您的 Google 帳號")
        print("3. 點擊 'Create API Key'")
        print("4. 複製生成的 API 金鑰")
        return False
    
    client = GeminiLLMClient(GOOGLE_API_KEY, GEMINI_MODEL)
    
    try:
        # 簡單的連線測試
        test_prompt = "請用繁體中文回答：你好，請告訴我你是什麼AI模型？請用JSON格式回傳：{\"model\": \"模型名稱\", \"language\": \"語言\"}"
        
        print(f"📡 使用模型: {GEMINI_MODEL}")
        print("🔄 發送測試請求...")
        
        start_time = time.time()
        response = await client.complete(test_prompt)
        response_time = int((time.time() - start_time) * 1000)
        
        print(f"✅ 連線成功！(響應時間: {response_time}ms)")
        print(f"📝 回應: {response[:200]}...")
        
        # 嘗試解析JSON
        try:
            result = json.loads(response)
            print(f"🎯 JSON 解析成功: {result}")
        except json.JSONDecodeError:
            print("⚠️ 回應不是有效的JSON格式，但連線正常")
        
        return True
        
    except Exception as e:
        print(f"❌ 連線失敗: {str(e)}")
        return False
        
    finally:
        await client.close()


async def test_gemini_product_extraction():
    """測試 Gemini 產品屬性提取"""
    
    print("\n🤖 測試 Gemini 產品屬性提取")
    print("=" * 50)
    
    if GOOGLE_API_KEY == 'your-google-api-key-here':
        return
    
    client = GeminiLLMClient(GOOGLE_API_KEY, GEMINI_MODEL)
    
    # 測試產品資料（繁體中文）
    test_products = [
        {
            "title": "iPhone 15 Pro Max 256GB 鈦藍色",
            "description": "蘋果最新旗艦手機，6.7吋超級視網膜XDR顯示器，A17 Pro仿生晶片，4800萬像素主相機，支援5G和MagSafe"
        },
        {
            "title": "Samsung Galaxy S24 Ultra 512GB 鈦黑色",
            "description": "三星頂級智慧型手機，6.8吋動態AMOLED 2X螢幕，Snapdragon 8 Gen 3處理器，內建S Pen手寫筆，2億像素相機"
        },
        {
            "title": "Google Pixel 8 Pro 128GB 瓷白色",
            "description": "Google最新AI手機，6.7吋LTPO OLED顯示器，Google Tensor G3晶片，Magic Eraser魔術橡皮擦功能"
        }
    ]
    
    try:
        for i, product in enumerate(test_products, 1):
            print(f"\n📱 測試產品 {i}: {product['title']}")
            
            # 構建繁體中文提取提示
            prompt = f"""
請從以下產品資訊中提取結構化屬性，用繁體中文回答：

產品標題: {product['title']}
產品描述: {product['description']}

請以 JSON 格式回應，包含以下欄位：
{{
    "brand": "品牌名稱",
    "model": "型號",
    "color": "顏色",
    "storage": "儲存容量",
    "screen_size": "螢幕尺寸",
    "processor": "處理器",
    "camera": "主要相機規格",
    "price_tier": "價格等級 (入門級/中階/高階/旗艦)",
    "category": "產品類別",
    "key_features": ["主要特色1", "主要特色2", "主要特色3"]
}}

請只回傳有效的 JSON 格式，不要包含其他文字或解釋。
"""
            
            start_time = time.time()
            
            try:
                # 調用 Gemini API
                response = await client.complete(prompt)
                processing_time = int((time.time() - start_time) * 1000)
                
                # 清理回應（移除可能的markdown標記）
                clean_response = response.strip()
                if clean_response.startswith('```json'):
                    clean_response = clean_response[7:]
                if clean_response.endswith('```'):
                    clean_response = clean_response[:-3]
                clean_response = clean_response.strip()
                
                # 解析JSON
                result = json.loads(clean_response)
                
                print(f"✅ 提取成功 (耗時: {processing_time}ms)")
                print(f"   品牌: {result.get('brand', 'N/A')}")
                print(f"   型號: {result.get('model', 'N/A')}")
                print(f"   顏色: {result.get('color', 'N/A')}")
                print(f"   容量: {result.get('storage', 'N/A')}")
                print(f"   螢幕: {result.get('screen_size', 'N/A')}")
                print(f"   處理器: {result.get('processor', 'N/A')}")
                print(f"   相機: {result.get('camera', 'N/A')}")
                print(f"   等級: {result.get('price_tier', 'N/A')}")
                print(f"   類別: {result.get('category', 'N/A')}")
                
                features = result.get('key_features', [])
                if features:
                    print(f"   特色: {', '.join(features[:3])}")
                
            except json.JSONDecodeError as e:
                print(f"❌ JSON 解析失敗: {str(e)}")
                print(f"原始回應: {response[:300]}...")
                
            except Exception as e:
                print(f"❌ 處理失敗: {str(e)}")
    
    except Exception as e:
        print(f"❌ 測試失敗: {str(e)}")
        
    finally:
        await client.close()


async def test_gemini_category_classification():
    """測試 Gemini 產品分類"""
    
    print("\n🏷️ 測試 Gemini 產品分類")
    print("-" * 30)
    
    if GOOGLE_API_KEY == 'your-google-api-key-here':
        return
    
    client = GeminiLLMClient(GOOGLE_API_KEY, GEMINI_MODEL)
    
    test_items = [
        "Nike Air Jordan 1 男款籃球鞋 黑紅配色",
        "Sony WH-1000XM5 無線降噪耳機 銀色",
        "Dyson V15 Detect 無線手持吸塵器",
        "美的 IH電磁爐 智能觸控面板 2100W",
        "Uniqlo Heattech極暖衣 男款長袖內衣",
        "YSL Rouge Pur Couture 奢華緞面唇膏"
    ]
    
    # 繁體中文類別
    categories = [
        "服裝配件", "鞋類", "電子產品", "家電用品", 
        "運動用品", "美容保養", "家居用品", "汽車用品",
        "3C數位", "時尚配件", "生活用品", "健康保健"
    ]
    
    try:
        for item in test_items:
            prompt = f"""
請將以下產品分類到最合適的類別中，用繁體中文回答：

產品: {item}

可用類別: {', '.join(categories)}

請回傳 JSON 格式:
{{
    "predicted_category": "最合適的類別",
    "confidence": 0.95,
    "reasoning": "選擇理由"
}}

請只回傳有效的 JSON 格式。
"""
            
            try:
                response = await client.complete(prompt)
                
                # 清理回應
                clean_response = response.strip()
                if clean_response.startswith('```json'):
                    clean_response = clean_response[7:]
                if clean_response.endswith('```'):
                    clean_response = clean_response[:-3]
                clean_response = clean_response.strip()
                
                result = json.loads(clean_response)
                category = result.get('predicted_category', 'N/A')
                confidence = result.get('confidence', 0.0)
                reasoning = result.get('reasoning', '')
                
                print(f"📦 {item}")
                print(f"   → {category} (信心度: {confidence:.2f})")
                if reasoning:
                    print(f"   理由: {reasoning}")
                
            except json.JSONDecodeError:
                print(f"❌ 分類失敗: {item}")
            except Exception as e:
                print(f"❌ 處理 {item} 時發生錯誤: {str(e)}")
    
    except Exception as e:
        print(f"❌ 分類測試失敗: {str(e)}")
        
    finally:
        await client.close()


async def main():
    """主測試函數"""
    
    print("🚀 Google Gemini API 整合測試")
    print("=" * 50)
    
    # 檢查依賴
    try:
        import aiohttp
    except ImportError:
        print("❌ 需要安裝 aiohttp: pip install aiohttp")
        return
    
    # 執行測試
    connection_ok = await test_gemini_connection()
    
    if connection_ok:
        await test_gemini_product_extraction()
        await test_gemini_category_classification()
        
        print("\n✅ Gemini 整合測試完成！")
        print("\n🎯 下一步：")
        print("1. 將 Gemini 客戶端整合到 Echo-Roots 系統")
        print("2. 調整提示模板以優化結果")
        print("3. 測試大量資料處理性能")
    else:
        print("\n❌ 請先解決連線問題再繼續")


if __name__ == "__main__":
    asyncio.run(main())
