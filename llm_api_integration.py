#!/usr/bin/env python3
"""
LLM API 整合範例
Examples of how to integrate your own LLM APIs with Echo-Roots
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

# 不同 LLM 提供商的客戶端
import openai  # pip install openai
import anthropic  # pip install anthropic
import requests  # 自訂 API


class BaseLLMClient(ABC):
    """LLM 客戶端基礎類別"""
    
    def __init__(self, api_key: str, model_name: str, **kwargs):
        self.api_key = api_key
        self.model_name = model_name
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    async def complete(self, prompt: str, **kwargs) -> str:
        """完成提示的抽象方法"""
        pass


class OpenAIClient(BaseLLMClient):
    """OpenAI GPT 客戶端"""
    
    def __init__(self, api_key: str, model_name: str = "gpt-4", **kwargs):
        super().__init__(api_key, model_name, **kwargs)
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.temperature = kwargs.get('temperature', 0.3)
        self.max_tokens = kwargs.get('max_tokens', 2000)
    
    async def complete(self, prompt: str, **kwargs) -> str:
        """使用 OpenAI API 完成提示"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a product data extraction expert. Return valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                response_format={"type": "json_object"}  # 確保返回 JSON
            )
            
            content = response.choices[0].message.content.strip()
            self.logger.info(f"OpenAI response received, length: {len(content)}")
            return content
            
        except Exception as e:
            self.logger.error(f"OpenAI API error: {str(e)}")
            raise


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude 客戶端"""
    
    def __init__(self, api_key: str, model_name: str = "claude-3-sonnet-20240229", **kwargs):
        super().__init__(api_key, model_name, **kwargs)
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.max_tokens = kwargs.get('max_tokens', 2000)
    
    async def complete(self, prompt: str, **kwargs) -> str:
        """使用 Anthropic API 完成提示"""
        try:
            response = await self.client.messages.create(
                model=self.model_name,
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                messages=[
                    {
                        "role": "user", 
                        "content": f"{prompt}\n\nPlease respond with valid JSON only."
                    }
                ]
            )
            
            content = response.content[0].text.strip()
            self.logger.info(f"Anthropic response received, length: {len(content)}")
            return content
            
        except Exception as e:
            self.logger.error(f"Anthropic API error: {str(e)}")
            raise


class CustomAPIClient(BaseLLMClient):
    """自訂 API 客戶端範例"""
    
    def __init__(self, api_key: str, base_url: str, model_name: str, **kwargs):
        super().__init__(api_key, model_name, **kwargs)
        self.base_url = base_url.rstrip('/')
        self.timeout = kwargs.get('timeout', 30)
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
    
    async def complete(self, prompt: str, **kwargs) -> str:
        """使用自訂 API 完成提示"""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "max_tokens": kwargs.get('max_tokens', 2000),
                "temperature": kwargs.get('temperature', 0.3),
                "format": "json"
            }
            
            # 使用 requests (同步) 或 aiohttp (異步)
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/v1/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data.get('choices', [{}])[0].get('text', '').strip()
                        self.logger.info(f"Custom API response received, length: {len(content)}")
                        return content
                    else:
                        raise Exception(f"API returned status {response.status}: {await response.text()}")
                        
        except Exception as e:
            self.logger.error(f"Custom API error: {str(e)}")
            raise


class AzureOpenAIClient(BaseLLMClient):
    """Azure OpenAI 客戶端"""
    
    def __init__(self, api_key: str, endpoint: str, deployment_name: str, api_version: str = "2024-02-15-preview", **kwargs):
        super().__init__(api_key, deployment_name, **kwargs)
        self.client = openai.AsyncAzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint
        )
        self.deployment_name = deployment_name
        self.temperature = kwargs.get('temperature', 0.3)
        self.max_tokens = kwargs.get('max_tokens', 2000)
    
    async def complete(self, prompt: str, **kwargs) -> str:
        """使用 Azure OpenAI API 完成提示"""
        try:
            response = await self.client.chat.completions.create(
                model=self.deployment_name,  # Azure 使用 deployment name
                messages=[
                    {"role": "system", "content": "You are a product data extraction expert. Return valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=kwargs.get('max_tokens', self.max_tokens)
            )
            
            content = response.choices[0].message.content.strip()
            self.logger.info(f"Azure OpenAI response received, length: {len(content)}")
            return content
            
        except Exception as e:
            self.logger.error(f"Azure OpenAI API error: {str(e)}")
            raise


class GoogleVertexAIClient(BaseLLMClient):
    """Google Vertex AI 客戶端"""
    
    def __init__(self, project_id: str, location: str, model_name: str = "text-bison", **kwargs):
        super().__init__("", model_name, **kwargs)  # Vertex AI 使用服務帳戶認證
        self.project_id = project_id
        self.location = location
        
        # 需要安裝 google-cloud-aiplatform
        try:
            from google.cloud import aiplatform
            aiplatform.init(project=project_id, location=location)
            self.aiplatform = aiplatform
        except ImportError:
            raise ImportError("Please install google-cloud-aiplatform: pip install google-cloud-aiplatform")
    
    async def complete(self, prompt: str, **kwargs) -> str:
        """使用 Google Vertex AI 完成提示"""
        try:
            from google.cloud.aiplatform.gapic.schema import predict
            
            # 建構請求
            instance = predict.instance.TextGenerationPredictInstance(
                prompt=prompt
            )
            
            parameters = predict.params.TextGenerationPredictParams(
                temperature=kwargs.get('temperature', 0.3),
                max_output_tokens=kwargs.get('max_tokens', 2000),
                top_p=kwargs.get('top_p', 0.8),
                top_k=kwargs.get('top_k', 40),
            )
            
            # 呼叫 API
            endpoint = self.aiplatform.Endpoint(
                endpoint_name=f"projects/{self.project_id}/locations/{self.location}/endpoints/{self.model_name}"
            )
            
            response = await endpoint.predict_async(
                instances=[instance],
                parameters=parameters
            )
            
            content = response.predictions[0]['content'].strip()
            self.logger.info(f"Vertex AI response received, length: {len(content)}")
            return content
            
        except Exception as e:
            self.logger.error(f"Vertex AI error: {str(e)}")
            raise


# =============================================================================
# 整合到 Echo-Roots 的範例
# =============================================================================

async def create_echo_roots_llm_extractor():
    """建立整合了自訂 LLM 的 Echo-Roots 提取器"""
    
    # 1. 選擇您的 LLM 客戶端
    # 選項 A: OpenAI
    llm_client = OpenAIClient(
        api_key="your-openai-api-key",
        model_name="gpt-4",
        temperature=0.3,
        max_tokens=2000
    )
    
    # 選項 B: Anthropic Claude
    # llm_client = AnthropicClient(
    #     api_key="your-anthropic-api-key",
    #     model_name="claude-3-sonnet-20240229",
    #     max_tokens=2000
    # )
    
    # 選項 C: Azure OpenAI
    # llm_client = AzureOpenAIClient(
    #     api_key="your-azure-api-key",
    #     endpoint="https://your-resource.openai.azure.com/",
    #     deployment_name="your-deployment-name",
    #     api_version="2024-02-15-preview"
    # )
    
    # 選項 D: 自訂 API
    # llm_client = CustomAPIClient(
    #     api_key="your-custom-api-key",
    #     base_url="https://your-api.com",
    #     model_name="your-model-name"
    # )
    
    # 2. 載入領域包
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    
    from echo_roots.models.domain import DomainPack
    from echo_roots.pipelines.extraction import LLMExtractor, ExtractorConfig
    import yaml
    
    # 載入電商領域包
    domain_path = Path("domains/ecommerce/domain.yaml")
    with open(domain_path, 'r', encoding='utf-8') as f:
        domain_config = yaml.safe_load(f)
    
    domain_pack = DomainPack(**domain_config)
    
    # 3. 設定提取器配置
    config = ExtractorConfig(
        model_name=llm_client.model_name,
        temperature=0.3,
        max_tokens=2000,
        enable_validation=True,
        retry_attempts=3,
        retry_delay=1.0
    )
    
    # 4. 建立提取器
    extractor = LLMExtractor(
        domain_pack=domain_pack,
        llm_client=llm_client,
        config=config
    )
    
    return extractor


async def test_real_llm_extraction():
    """測試真實 LLM 提取"""
    try:
        # 建立提取器
        extractor = await create_echo_roots_llm_extractor()
        
        # 載入測試資料
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        
        from echo_roots.storage.duckdb_backend import DuckDBStorageManager
        
        storage_config = {
            "duckdb": {
                "database_path": "./test_echo_roots.db"
            }
        }
        
        storage_manager = DuckDBStorageManager(storage_config)
        await storage_manager.initialize()
        
        # 獲取測試項目
        items = await storage_manager.ingestion.list_items(limit=1)
        if not items:
            print("❌ No test data found. Please run test_data_processing.py first.")
            return
        
        # 測試真實 LLM 提取
        test_item = items[0]
        print(f"🤖 Testing real LLM extraction for: {test_item.title}")
        
        result = await extractor.extract_single(test_item)
        
        print(f"✅ Extracted {len(result.attributes)} attributes:")
        for attr in result.attributes:
            print(f"  - {attr.name}: {attr.value} (confidence: {attr.confidence:.2f})")
            print(f"    Evidence: {attr.evidence}")
        
        print(f"✅ Extracted {len(result.terms)} terms:")
        for term in result.terms[:5]:
            print(f"  - {term.term} (confidence: {term.confidence:.2f})")
        
        print(f"⏱️ Processing time: {result.metadata.processing_time_ms}ms")
        print(f"🔧 Model used: {result.metadata.model}")
        
        await storage_manager.close()
        
    except Exception as e:
        print(f"❌ Error testing real LLM extraction: {str(e)}")
        import traceback
        traceback.print_exc()


# =============================================================================
# 配置檔案範例
# =============================================================================

def create_llm_config_file():
    """建立 LLM 配置檔案範例"""
    
    config = {
        "llm": {
            "provider": "openai",  # openai, anthropic, azure, custom, vertex
            "model_name": "gpt-4",
            "api_key": "${OPENAI_API_KEY}",  # 使用環境變數
            "temperature": 0.3,
            "max_tokens": 2000,
            "timeout": 30,
            
            # Azure 專用設定
            "azure": {
                "endpoint": "${AZURE_OPENAI_ENDPOINT}",
                "deployment_name": "${AZURE_DEPLOYMENT_NAME}",
                "api_version": "2024-02-15-preview"
            },
            
            # 自訂 API 設定
            "custom": {
                "base_url": "${CUSTOM_API_URL}",
                "headers": {
                    "Authorization": "Bearer ${CUSTOM_API_KEY}",
                    "Custom-Header": "value"
                }
            },
            
            # Google Vertex AI 設定
            "vertex": {
                "project_id": "${GOOGLE_PROJECT_ID}",
                "location": "us-central1",
                "service_account_path": "${GOOGLE_SERVICE_ACCOUNT_JSON}"
            }
        },
        
        "extraction": {
            "enable_validation": True,
            "retry_attempts": 3,
            "retry_delay": 1.0,
            "batch_size": 10,
            "concurrent_requests": 5
        }
    }
    
    import yaml
    with open("llm_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print("✅ Created llm_config.yaml")
    print("📝 Please update the configuration with your actual API keys and settings")


if __name__ == "__main__":
    # 建立配置檔案範例
    create_llm_config_file()
    
    # 測試真實 LLM 提取（需要先設定 API 金鑰）
    # asyncio.run(test_real_llm_extraction())
