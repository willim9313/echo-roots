"""
vLLM Integration Demo for echo-roots.

This script demonstrates how to integrate vLLM with echo-roots extraction pipeline.
It provides examples for different deployment scenarios and use cases.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from echo_roots.models.core import IngestionItem
from echo_roots.pipelines.llm_factory import LLMClientFactory


class VLLMDemo:
    """Demo class for vLLM integration."""
    
    def __init__(self):
        self.examples = []
    
    def add_example(self, name: str, description: str, config: dict, test_prompt: str = None):
        """Add a configuration example."""
        self.examples.append({
            "name": name,
            "description": description,
            "config": config,
            "test_prompt": test_prompt or "What is the capital of Taiwan? Answer in Chinese."
        })
    
    def setup_examples(self):
        """Setup various vLLM configuration examples."""
        
        # Local deployment with default settings
        self.add_example(
            "Local Default",
            "本地部署，使用預設設定",
            {
                "vllm": {
                    "deployment_type": "local",
                    "model_name": "meta-llama/Llama-2-7b-chat-hf"
                }
            }
        )
        
        # Local deployment with custom settings
        self.add_example(
            "Local Custom",
            "本地部署，自定義端口和參數",
            {
                "vllm": {
                    "deployment_type": "local",
                    "host": "localhost",
                    "port": 8001,
                    "model_name": "Qwen/Qwen-7B-Chat",
                    "timeout": 90
                }
            },
            "請用中文回答：台灣的首都是哪裡？"
        )
        
        # Cloud deployment
        self.add_example(
            "Cloud Deployment",
            "雲端部署，使用遠程 vLLM 服務",
            {
                "vllm": {
                    "deployment_type": "cloud",
                    "base_url": "https://your-vllm-server.com/v1",
                    "model_name": "mistralai/Mixtral-8x7B-Instruct-v0.1",
                    "api_key": "your-api-key",
                    "timeout": 120
                }
            }
        )
        
        # Generic deployment
        self.add_example(
            "Generic",
            "通用配置，適用於任何 OpenAI 相容端點",
            {
                "vllm": {
                    "base_url": "http://custom-host:9000/v1",
                    "model_name": "custom-model",
                    "api_key": "dummy-key"
                }
            }
        )
    
    def print_configuration_guide(self):
        """Print configuration guide."""
        print("🔧 vLLM Configuration Guide")
        print("=" * 50)
        
        print("\\n📋 Available Configuration Examples:\\n")
        
        for i, example in enumerate(self.examples, 1):
            print(f"{i}. **{example['name']}**")
            print(f"   描述: {example['description']}")
            print(f"   配置: {example['config']}")
            print(f"   測試提示: {example['test_prompt']}")
            print()
    
    def print_server_setup_guide(self):
        """Print vLLM server setup guide."""
        print("\\n🚀 vLLM Server Setup Commands")
        print("=" * 40)
        
        setups = [
            {
                "name": "Llama-2 7B (英文通用)",
                "command": """python -m vllm.entrypoints.openai.api_server \\
    --model meta-llama/Llama-2-7b-chat-hf \\
    --port 8000 \\
    --host 0.0.0.0"""
            },
            {
                "name": "Qwen 7B (中文優化)", 
                "command": """python -m vllm.entrypoints.openai.api_server \\
    --model Qwen/Qwen-7B-Chat \\
    --port 8001 \\
    --trust-remote-code"""
            },
            {
                "name": "Mixtral 8x7B (高效能)",
                "command": """python -m vllm.entrypoints.openai.api_server \\
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \\
    --port 8002 \\
    --tensor-parallel-size 2"""
            }
        ]
        
        for setup in setups:
            print(f"\\n📦 {setup['name']}:")
            print(f"```bash")
            print(f"{setup['command']}")
            print(f"```")
    
    def print_environment_variables(self):
        """Print environment variable examples."""
        print("\\n🔐 Environment Variables")
        print("=" * 35)
        
        envs = [
            ("VLLM_BASE_URL", "http://localhost:8000/v1", "vLLM 服務器 URL"),
            ("VLLM_MODEL_NAME", "meta-llama/Llama-2-7b-chat-hf", "模型名稱"),
            ("VLLM_API_KEY", "dummy-key", "API 金鑰 (本地通常不需要)"),
            ("VLLM_DEPLOYMENT_TYPE", "local", "部署類型 (local/cloud)")
        ]
        
        print("\\n設定範例:")
        for env_name, example_value, description in envs:
            print(f"export {env_name}='{example_value}'  # {description}")
    
    async def test_configuration_example(self, example_name: str):
        """Test a specific configuration example."""
        example = next((ex for ex in self.examples if ex["name"] == example_name), None)
        if not example:
            print(f"❌ Example '{example_name}' not found")
            return False
        
        print(f"\\n🧪 Testing: {example['name']}")
        print(f"📝 Description: {example['description']}")
        
        try:
            # Create client
            client = LLMClientFactory.create_client("vllm", example["config"])
            print(f"✅ Client created: {type(client).__name__}")
            print(f"🔗 URL: {getattr(client, 'base_url', 'N/A')}")
            print(f"🤖 Model: {getattr(client, 'model_name', 'N/A')}")
            
            # Test health check if available
            if hasattr(client, 'health_check'):
                try:
                    healthy = await asyncio.wait_for(client.health_check(), timeout=10)
                    if healthy:
                        print("✅ Health check passed")
                        
                        # Test completion
                        response = await asyncio.wait_for(
                            client.complete(
                                example["test_prompt"],
                                max_tokens=100,
                                temperature=0.3
                            ),
                            timeout=30
                        )
                        print(f"✅ Test completion successful")
                        print(f"📄 Response: {response[:100]}...")
                        return True
                        
                    else:
                        print("⚠️  Health check failed - server not responding")
                        
                except asyncio.TimeoutError:
                    print("⚠️  Health check timeout - server may not be running")
                except Exception as e:
                    print(f"⚠️  Health check error: {e}")
            
            print("💡 Client created successfully but server not tested")
            return True
            
        except Exception as e:
            print(f"❌ Configuration test failed: {e}")
            return False
    
    async def demo_extraction_pipeline(self):
        """Demonstrate extraction pipeline with vLLM."""
        print("\\n\\n🔄 Extraction Pipeline Demo")
        print("=" * 40)
        
        # Try to use any available vLLM configuration
        test_configs = [
            ("Local Default", {"vllm": {"deployment_type": "local", "model_name": "meta-llama/Llama-2-7b-chat-hf"}}),
            ("Environment", {})  # Use environment variables
        ]
        
        working_client = None
        
        for config_name, config in test_configs:
            try:
                print(f"\\n🔧 Trying {config_name} configuration...")
                client = LLMClientFactory.create_client("vllm", config)
                
                if hasattr(client, 'health_check'):
                    healthy = await asyncio.wait_for(client.health_check(), timeout=5)
                    if healthy:
                        working_client = client
                        print(f"✅ Using {config_name} configuration")
                        break
                    else:
                        print(f"⚠️  {config_name} server not responding")
                else:
                    # Can't test health, but client was created
                    working_client = client
                    print(f"⚠️  {config_name} client created (server not tested)")
                    break
                    
            except Exception as e:
                print(f"❌ {config_name} failed: {e}")
        
        if not working_client:
            print("\\n❌ No working vLLM configuration found")
            print("💡 Make sure vLLM server is running and environment variables are set")
            return False
        
        # Create test items
        print("\\n📝 Creating test items...")
        test_items = [
            IngestionItem(
                item_id="demo_001",
                title="Apple iPhone 15 Pro Max 1TB",
                description="最新的 iPhone 旗艦機型，搭載 A17 Pro 晶片，鈦金屬機身，支援 USB-C",
                language="zh",
                source="vllm_demo"
            ),
            IngestionItem(
                item_id="demo_002", 
                title="Samsung Galaxy S24 Ultra 512GB",
                description="Samsung flagship smartphone with S Pen, 200MP camera, and AI features",
                language="en",
                source="vllm_demo"
            )
        ]
        
        print(f"✅ Created {len(test_items)} test items")
        
        # Simple extraction test (without full domain configuration)
        print("\\n🔬 Testing simple extraction...")
        
        for item in test_items:
            try:
                prompt = f"""Extract product information from the following item:
Title: {item.title}
Description: {item.description}

Return JSON with extracted attributes in this format:
{{"brand": "extracted_brand", "product_type": "extracted_type", "model": "extracted_model"}}"""

                response = await asyncio.wait_for(
                    working_client.complete(prompt, max_tokens=200, temperature=0.1),
                    timeout=30
                )
                
                print(f"\\n📋 Item: {item.title}")
                print(f"🤖 Extraction: {response[:150]}...")
                
            except Exception as e:
                print(f"❌ Extraction failed for {item.item_id}: {e}")
        
        return True


async def main():
    """Main demo function."""
    print("🚀 vLLM Integration Demo for echo-roots")
    print("=" * 60)
    
    demo = VLLMDemo()
    demo.setup_examples()
    
    # Show configuration guide
    demo.print_configuration_guide()
    demo.print_server_setup_guide() 
    demo.print_environment_variables()
    
    # Check current environment
    print("\\n\\n🔍 Current Environment Status")
    print("=" * 40)
    
    vllm_vars = [
        ("VLLM_BASE_URL", os.getenv("VLLM_BASE_URL")),
        ("VLLM_MODEL_NAME", os.getenv("VLLM_MODEL_NAME")),
        ("VLLM_API_KEY", os.getenv("VLLM_API_KEY")),
        ("VLLM_DEPLOYMENT_TYPE", os.getenv("VLLM_DEPLOYMENT_TYPE"))
    ]
    
    for var_name, var_value in vllm_vars:
        if var_value:
            display_value = var_value if "API_KEY" not in var_name else var_value[:4] + "***" + var_value[-4:]
            print(f"✅ {var_name}: {display_value}")
        else:
            print(f"❌ {var_name}: Not set")
    
    # Provider availability check
    print("\\n🔌 Provider Availability Check")
    print("-" * 35)
    
    from echo_roots.pipelines.llm_factory import get_available_providers
    providers = get_available_providers()
    
    for provider, (available, status) in providers.items():
        symbol = "✅" if available else "❌"
        print(f"{symbol} {provider}: {status}")
    
    # Interactive testing
    print("\\n\\n🎯 Interactive Testing")
    print("=" * 30)
    
    if len(sys.argv) > 1:
        example_name = sys.argv[1]
        print(f"Testing specific example: {example_name}")
        await demo.test_configuration_example(example_name)
    else:
        print("Available test examples:")
        for i, example in enumerate(demo.examples, 1):
            print(f"  {i}. {example['name']}")
        
        print("\\n💡 Usage:")
        print(f"  python {sys.argv[0]} 'Local Default'")
        print(f"  python {sys.argv[0]} 'Local Custom'")
    
    # Try extraction demo
    await demo.demo_extraction_pipeline()
    
    print("\\n\\n📚 Next Steps")
    print("=" * 20)
    print("1. 啟動 vLLM 伺服器 (參考上面的命令)")
    print("2. 設定環境變數")
    print("3. 執行完整測試: python test_vllm_integration.py")
    print("4. 查看文檔: docs/guides/vllm-integration.md")


if __name__ == "__main__":
    asyncio.run(main())
