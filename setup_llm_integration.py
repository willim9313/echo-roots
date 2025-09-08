#!/usr/bin/env python3
"""
實際整合 LLM API 到 Echo-Roots 的步驟指南
Step-by-step guide to integrate your LLM API into Echo-Roots
"""

# ==============================================================================
# 步驟 1: 在現有的 extraction.py 中添加您的 LLM 客戶端
# ==============================================================================

"""
您需要在 src/echo_roots/pipelines/extraction.py 中添加新的 LLM 客戶端類別。

找到 MockLLMClient 類別後面，添加以下代碼：
"""

class YourCustomLLMClient:
    """您的自訂 LLM 客戶端"""
    
    def __init__(self, api_key: str, model_name: str = "gpt-4", **kwargs):
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = kwargs.get('base_url', 'https://api.openai.com/v1')
        self.temperature = kwargs.get('temperature', 0.3)
        self.max_tokens = kwargs.get('max_tokens', 2000)
        
        # 設定 HTTP 客戶端
        import aiohttp
        self.session = None
    
    async def complete(self, prompt: str, **kwargs) -> str:
        """實現 LLMClient Protocol 的 complete 方法"""
        import aiohttp
        import json
        
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
                    raise Exception(f"API Error {response.status}: {error_text}")
                    
        except Exception as e:
            raise Exception(f"LLM API call failed: {str(e)}")
    
    async def close(self):
        """關閉 HTTP 會話"""
        if self.session:
            await self.session.close()


# ==============================================================================
# 步驟 2: 創建 LLM 客戶端工廠函數
# ==============================================================================

def create_llm_client(provider: str, **config) -> 'LLMClient':
    """根據配置創建 LLM 客戶端"""
    
    if provider == "openai":
        return YourCustomLLMClient(
            api_key=config['api_key'],
            model_name=config.get('model_name', 'gpt-4'),
            base_url=config.get('base_url', 'https://api.openai.com/v1'),
            temperature=config.get('temperature', 0.3),
            max_tokens=config.get('max_tokens', 2000)
        )
    
    elif provider == "anthropic":
        # 實現 Anthropic 客戶端
        class AnthropicLLMClient:
            def __init__(self, api_key: str, model_name: str = "claude-3-sonnet-20240229", **kwargs):
                self.api_key = api_key
                self.model_name = model_name
                self.max_tokens = kwargs.get('max_tokens', 2000)
            
            async def complete(self, prompt: str, **kwargs) -> str:
                import aiohttp
                import json
                
                headers = {
                    'x-api-key': self.api_key,
                    'Content-Type': 'application/json',
                    'anthropic-version': '2023-06-01'
                }
                
                payload = {
                    'model': self.model_name,
                    'max_tokens': kwargs.get('max_tokens', self.max_tokens),
                    'messages': [
                        {
                            'role': 'user',
                            'content': f'{prompt}\n\nPlease respond with valid JSON only.'
                        }
                    ]
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        'https://api.anthropic.com/v1/messages',
                        headers=headers,
                        json=payload,
                        timeout=30
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            return data['content'][0]['text'].strip()
                        else:
                            error_text = await response.text()
                            raise Exception(f"Anthropic API Error {response.status}: {error_text}")
        
        return AnthropicLLMClient(
            api_key=config['api_key'],
            model_name=config.get('model_name', 'claude-3-sonnet-20240229'),
            max_tokens=config.get('max_tokens', 2000)
        )
    
    elif provider == "custom":
        # 您的自訂 API 客戶端
        class CustomAPIClient:
            def __init__(self, **config):
                self.api_key = config['api_key']
                self.base_url = config['base_url']
                self.model_name = config['model_name']
                self.headers = config.get('headers', {})
            
            async def complete(self, prompt: str, **kwargs) -> str:
                import aiohttp
                
                headers = {
                    **self.headers,
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json'
                }
                
                # 根據您的 API 格式調整 payload
                payload = {
                    'model': self.model_name,
                    'prompt': prompt,
                    'max_tokens': kwargs.get('max_tokens', 2000),
                    'temperature': kwargs.get('temperature', 0.3)
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f'{self.base_url}/completions',
                        headers=headers,
                        json=payload,
                        timeout=30
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            # 根據您的 API 響應格式調整
                            return data['choices'][0]['text'].strip()
                        else:
                            error_text = await response.text()
                            raise Exception(f"Custom API Error {response.status}: {error_text}")
        
        return CustomAPIClient(**config)
    
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


# ==============================================================================
# 步驟 3: 創建配置文件
# ==============================================================================

import yaml
import os
from pathlib import Path

def create_llm_config():
    """創建 LLM 配置文件"""
    
    config = {
        'llm': {
            'provider': 'openai',  # 或 'anthropic', 'custom'
            'api_key': os.getenv('OPENAI_API_KEY', 'your-api-key-here'),
            'model_name': 'gpt-4',
            'temperature': 0.3,
            'max_tokens': 2000,
            'base_url': 'https://api.openai.com/v1',  # 可選，用於 API 代理
            
            # 如果使用 Anthropic
            'anthropic': {
                'api_key': os.getenv('ANTHROPIC_API_KEY', 'your-anthropic-key'),
                'model_name': 'claude-3-sonnet-20240229'
            },
            
            # 如果使用自訂 API
            'custom': {
                'api_key': os.getenv('CUSTOM_API_KEY', 'your-custom-key'),
                'base_url': 'https://your-api.com/v1',
                'model_name': 'your-model-name',
                'headers': {
                    'Custom-Header': 'value'
                }
            }
        },
        
        'extraction': {
            'retry_attempts': 3,
            'retry_delay': 1.0,
            'timeout': 30,
            'enable_validation': True
        }
    }
    
    config_path = Path('config/llm_config.yaml')
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"✅ 已創建配置文件: {config_path}")
    return config_path


# ==============================================================================
# 步驟 4: 修改 Echo-Roots 以使用您的 LLM
# ==============================================================================

def create_echo_roots_llm_integration():
    """創建整合腳本"""
    
    integration_code = '''
import asyncio
import yaml
from pathlib import Path
import sys

# 添加 src 到路徑
sys.path.insert(0, str(Path(__file__).parent / "src"))

from echo_roots.models.domain import DomainPack
from echo_roots.pipelines.extraction import LLMExtractor, ExtractorConfig
from echo_roots.storage.duckdb_backend import DuckDBStorageManager

# 導入您的 LLM 客戶端創建函數
from llm_api_integration import create_llm_client

async def run_extraction_with_your_llm():
    """使用您的 LLM API 運行提取"""
    
    # 1. 載入配置
    with open('config/llm_config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 2. 創建 LLM 客戶端
    llm_config = config['llm']
    llm_client = create_llm_client(
        provider=llm_config['provider'],
        **llm_config
    )
    
    # 3. 載入領域包
    domain_path = Path("domains/ecommerce/domain.yaml")
    with open(domain_path, 'r', encoding='utf-8') as f:
        domain_config = yaml.safe_load(f)
    
    domain_pack = DomainPack(**domain_config)
    
    # 4. 創建提取器
    extractor_config = ExtractorConfig(
        model_name=llm_config['model_name'],
        temperature=llm_config['temperature'],
        max_tokens=llm_config['max_tokens'],
        retry_attempts=config['extraction']['retry_attempts'],
        retry_delay=config['extraction']['retry_delay'],
        enable_validation=config['extraction']['enable_validation']
    )
    
    extractor = LLMExtractor(
        domain_pack=domain_pack,
        llm_client=llm_client,
        config=extractor_config
    )
    
    # 5. 載入測試資料
    storage_config = {
        "duckdb": {
            "database_path": "./test_echo_roots.db"
        }
    }
    
    storage_manager = DuckDBStorageManager(storage_config)
    await storage_manager.initialize()
    
    # 6. 獲取要處理的項目
    items = await storage_manager.ingestion.list_items(limit=3)
    
    if not items:
        print("❌ 沒有找到測試資料。請先運行 test_data_processing.py")
        return
    
    # 7. 運行提取
    print(f"🤖 使用 {llm_config['provider']} {llm_config['model_name']} 進行提取...")
    
    for i, item in enumerate(items, 1):
        print(f"\\n📝 處理項目 {i}: {item.title}")
        
        try:
            result = await extractor.extract_single(item)
            
            print(f"✅ 提取了 {len(result.attributes)} 個屬性:")
            for attr in result.attributes:
                print(f"  - {attr.name}: {attr.value} (信心度: {attr.confidence:.2f})")
            
            print(f"✅ 提取了 {len(result.terms)} 個術語:")
            for term in result.terms[:3]:
                print(f"  - {term.term} (信心度: {term.confidence:.2f})")
            
            print(f"⏱️ 處理時間: {result.metadata.processing_time_ms}ms")
            
        except Exception as e:
            print(f"❌ 提取失敗: {str(e)}")
    
    # 8. 清理
    await storage_manager.close()
    
    # 如果 LLM 客戶端有清理方法，調用它
    if hasattr(llm_client, 'close'):
        await llm_client.close()
    
    print("\\n🎉 提取完成！")

if __name__ == "__main__":
    asyncio.run(run_extraction_with_your_llm())
'''
    
    with open('run_with_your_llm.py', 'w', encoding='utf-8') as f:
        f.write(integration_code)
    
    print("✅ 已創建 run_with_your_llm.py")


# ==============================================================================
# 步驟 5: 環境變數設定範例
# ==============================================================================

def create_env_example():
    """創建環境變數範例文件"""
    
    env_content = '''# LLM API 配置
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
CUSTOM_API_KEY=your_custom_api_key_here

# Azure OpenAI (如果使用)
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_DEPLOYMENT_NAME=your-deployment-name
AZURE_API_KEY=your-azure-api-key

# Google Cloud (如果使用)
GOOGLE_PROJECT_ID=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

# 自訂設定
CUSTOM_API_URL=https://your-api.com
CUSTOM_MODEL_NAME=your-model-name
'''
    
    with open('.env.example', 'w', encoding='utf-8') as f:
        f.write(env_content)
    
    print("✅ 已創建 .env.example")
    print("📝 請複製到 .env 並填入您的實際 API 金鑰")


# ==============================================================================
# 主要函數
# ==============================================================================

def main():
    """設置 LLM API 整合"""
    
    print("🔧 設置 Echo-Roots LLM API 整合...")
    
    # 創建必要的文件
    config_path = create_llm_config()
    create_echo_roots_llm_integration()
    create_env_example()
    
    print(f"""
🎉 LLM API 整合設置完成！

下一步:
1. 編輯 {config_path} 設置您的 LLM 提供商和模型
2. 複製 .env.example 到 .env 並填入 API 金鑰
3. 運行: python run_with_your_llm.py

支援的 LLM 提供商:
- OpenAI (gpt-4, gpt-3.5-turbo)
- Anthropic (claude-3-sonnet, claude-3-haiku)
- Azure OpenAI
- 自訂 API

需要安裝的套件:
pip install aiohttp
pip install openai  # 如果使用 OpenAI
pip install anthropic  # 如果使用 Anthropic
""")


if __name__ == "__main__":
    main()
