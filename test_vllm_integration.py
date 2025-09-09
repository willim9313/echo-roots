"""
Test vLLM integration with echo-roots extraction pipeline.

This script validates that the vLLM client works properly with the
extraction pipeline using OpenAI-compatible API endpoints.
"""

import asyncio
import os
from pathlib import Path

# Add src to path for imports
import sys
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from echo_roots.models.core import IngestionItem
from echo_roots.pipelines.llm_factory import LLMClientFactory, get_available_providers
from echo_roots.pipelines.vllm_client import VLLMLocalClient, VLLMCloudClient


async def test_vllm_availability():
    """Test vLLM provider availability."""
    
    print("🔍 Testing vLLM Provider Availability")
    print("=" * 45)
    
    # Check all providers
    providers = get_available_providers()
    
    print("\\nAll provider status:")
    for provider, (available, message) in providers.items():
        status = "✅" if available else "❌"
        print(f"   {status} {provider}: {message}")
    
    # Focus on vLLM
    vllm_available, vllm_status = providers.get("vllm", (False, "Not found"))
    
    if not vllm_available:
        print(f"\\n❌ vLLM not available: {vllm_status}")
        print("\\n💡 To test vLLM integration:")
        print("   1. Start vLLM server:")
        print("      python -m vllm.entrypoints.openai.api_server \\\\")
        print("        --model meta-llama/Llama-2-7b-chat-hf \\\\")
        print("        --port 8000")
        print("   2. Set environment variables:")
        print("      export VLLM_BASE_URL=http://localhost:8000/v1")
        print("      export VLLM_MODEL_NAME=meta-llama/Llama-2-7b-chat-hf")
        print("   3. Run this test again")
        return False
    
    print(f"\\n✅ vLLM is available: {vllm_status}")
    return True


async def test_vllm_client_direct():
    """Test vLLM client directly."""
    
    print("\\n\\n🧪 Testing vLLM Client Direct API")
    print("=" * 40)
    
    # Test configuration options
    test_configs = [
        {
            "name": "Local Default",
            "config": {
                "deployment_type": "local",
                "model_name": "meta-llama/Llama-2-7b-chat-hf"
            }
        },
        {
            "name": "Custom Local",
            "config": {
                "deployment_type": "local", 
                "model_name": "meta-llama/Llama-2-7b-chat-hf",
                "host": "localhost",
                "port": 8000
            }
        }
    ]
    
    # Add cloud config if URL is provided
    if os.getenv("VLLM_CLOUD_URL"):
        test_configs.append({
            "name": "Cloud",
            "config": {
                "deployment_type": "cloud",
                "base_url": os.getenv("VLLM_CLOUD_URL"),
                "model_name": os.getenv("VLLM_CLOUD_MODEL", "meta-llama/Llama-2-7b-chat-hf"),
                "api_key": os.getenv("VLLM_CLOUD_API_KEY", "dummy-key")
            }
        })
    
    success_count = 0
    
    for test_config in test_configs:
        print(f"\\n📋 Testing {test_config['name']} configuration...")
        
        try:
            # Create client using factory
            client = LLMClientFactory.create_client("vllm", {"vllm": test_config["config"]})
            print(f"   ✅ Client created: {type(client).__name__}")
            
            # Test health check
            healthy = await client.health_check()
            if healthy:
                print("   ✅ Health check passed")
            else:
                print("   ⚠️  Health check failed - server might not be running")
                continue
            
            # Test simple completion
            response = await client.complete(
                "What is the capital of France? Answer in one word.",
                max_tokens=10,
                temperature=0.1
            )
            print(f"   ✅ Completion test: '{response.strip()}'")
            
            # Test model info
            model_info = await client.get_model_info()
            print(f"   📊 Model info: {model_info.get('id', 'unknown')}")
            
            success_count += 1
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    print(f"\\n📈 Success rate: {success_count}/{len(test_configs)}")
    return success_count > 0


async def test_vllm_extraction_pipeline():
    """Test vLLM with extraction pipeline."""
    
    print("\\n\\n🔧 Testing vLLM with Extraction Pipeline")
    print("=" * 50)
    
    # Create test domain pack
    print("\\n1. Creating test domain pack...")
    from echo_roots.models.domain import DomainPack
    
    test_domain = DomainPack(
        domain="electronics",
        taxonomy_version="2025.01",
        input_mapping={
            "title": ["title", "name"],
            "description": ["description", "desc"],
            "language": ["language", "lang"],
            "source_uri": ["url", "uri"],
            "collected_at": ["timestamp", "collected_at"]
        },
        output_schema={
            "core_item": {
                "id": "str",
                "title": "str", 
                "description": "str",
                "language": "str"
            },
            "attributes": [
                {"key": "brand", "type": "text"},
                {"key": "category", "type": "text"},
                {"key": "model", "type": "text"}
            ]
        },
        attribute_hints={
            "brand": {
                "examples": ["Apple", "Samsung", "Google"],
                "notes": "Manufacturer brand name"
            },
            "category": {
                "examples": ["smartphone", "laptop", "tablet"],  
                "notes": "Product category"
            }
        }
    )
    print(f"   ✅ Domain pack created: {test_domain.domain}")
    
    # Create test item
    print("\\n2. Creating test item...")
    test_item = IngestionItem(
        item_id="vllm_test_001",
        title="Apple iPhone 15 Pro Max 256GB",
        description="最新的 iPhone 旗艦機型，搭載 A17 Pro 處理器，鈦金屬機身，支援 USB-C 接口",
        language="zh",
        source="vllm_test"
    )
    print(f"   📱 Test item: {test_item.title}")
    
    # Create vLLM client
    print("\\n3. Creating vLLM client...")
    try:
        client = LLMClientFactory.create_client("vllm")
        print(f"   ✅ vLLM client created: {client}")
        
        # Quick health check
        if hasattr(client, 'health_check'):
            healthy = await client.health_check()
            if not healthy:
                print("   ⚠️  Warning: vLLM server health check failed")
                return False
    
    except Exception as e:
        print(f"   ❌ Failed to create vLLM client: {e}")
        return False
    
    # Test extraction
    print("\\n4. Testing extraction...")
    try:
        from echo_roots.pipelines.extraction import LLMExtractor, ExtractorConfig
        
        config = ExtractorConfig(
            enable_validation=False,  # Disable for testing
            temperature=0.1,
            max_tokens=1500
        )
        
        extractor = LLMExtractor(
            domain_pack=test_domain,
            llm_client=client,
            config=config
        )
        
        result = await extractor.extract_single(test_item)
        
        print(f"   ✅ Extraction completed for: {result.item_id}")
        print(f"   📊 Extracted {len(result.attributes)} attributes and {len(result.terms)} terms")
        
        # Display results
        if result.attributes:
            print("\\n   📋 Extracted Attributes:")
            for attr in result.attributes:
                print(f"      - {attr.name}: '{attr.value}'")
                if attr.evidence:
                    print(f"        Evidence: {attr.evidence[:60]}...")
        
        if result.terms:
            print("\\n   🏷️  Extracted Terms:")
            for term in result.terms:
                print(f"      - {term.term} (confidence: {term.confidence:.2f})")
        
        print(f"\\n   ⏱️  Processing time: {result.metadata.processing_time_ms}ms")
        print(f"   🤖 Model used: {result.metadata.model}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_vllm_batch_processing():
    """Test vLLM batch processing capabilities."""
    
    print("\\n\\n📦 Testing vLLM Batch Processing")  
    print("=" * 40)
    
    try:
        client = LLMClientFactory.create_client("vllm")
        
        # Test batch completion
        prompts = [
            "What is 2+2? Answer with just the number.",
            "What is the color of the sky? Answer in one word.",
            "What is the capital of Japan? Answer in one word."
        ]
        
        print(f"\\n📝 Processing {len(prompts)} prompts in batch...")
        start_time = asyncio.get_event_loop().time()
        
        responses = await client.generate_batch(prompts, max_tokens=20, temperature=0.1)
        
        end_time = asyncio.get_event_loop().time()
        total_time = (end_time - start_time) * 1000
        
        print(f"   ✅ Batch completed in {total_time:.0f}ms")
        print(f"   📋 Results:")
        
        for i, (prompt, response) in enumerate(zip(prompts, responses), 1):
            print(f"      {i}. Q: {prompt[:30]}...")
            print(f"         A: {response.strip()}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Batch processing failed: {e}")
        return False


def main():
    """Main test function."""
    
    print("🚀 Echo-Roots vLLM Integration Test")
    print("=" * 50)
    
    print(f"Python: {sys.version}")
    print(f"Working directory: {Path.cwd()}")
    
    # Check environment
    print("\\n🔧 Environment Configuration:")
    vllm_vars = {
        "VLLM_BASE_URL": os.getenv("VLLM_BASE_URL"),
        "VLLM_MODEL_NAME": os.getenv("VLLM_MODEL_NAME"), 
        "VLLM_API_KEY": os.getenv("VLLM_API_KEY"),
        "VLLM_DEPLOYMENT_TYPE": os.getenv("VLLM_DEPLOYMENT_TYPE")
    }
    
    for var, value in vllm_vars.items():
        if value:
            if "API_KEY" in var and len(value) > 8:
                display_value = value[:4] + "*" * (len(value) - 8) + value[-4:]
            else:
                display_value = value
            print(f"   {var}: {display_value}")
        else:
            print(f"   {var}: Not set")
    
    async def run_tests():
        tests = [
            ("Provider Availability", test_vllm_availability()),
            ("Direct Client API", test_vllm_client_direct()),
            ("Extraction Pipeline", test_vllm_extraction_pipeline()),
            ("Batch Processing", test_vllm_batch_processing())
        ]
        
        results = {}
        
        for test_name, test_coro in tests:
            try:
                results[test_name] = await test_coro
            except Exception as e:
                print(f"\\n❌ Test '{test_name}' failed with exception: {e}")
                results[test_name] = False
        
        # Summary
        print("\\n\\n📋 Test Summary")
        print("=" * 30)
        
        for test_name, success in results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{test_name:<25} {status}")
        
        passed = sum(results.values())
        total = len(results)
        
        if passed == total:
            print(f"\\n✅ All {total} tests passed! vLLM integration is working correctly.")
            return 0
        elif passed > 0:
            print(f"\\n⚠️  {passed}/{total} tests passed. Some functionality may be limited.")
            return 1
        else:
            print(f"\\n❌ All tests failed. Check vLLM server and configuration.")
            return 2
    
    result = asyncio.run(run_tests())
    sys.exit(result)


if __name__ == "__main__":
    main()
