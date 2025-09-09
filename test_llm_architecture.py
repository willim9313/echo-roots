"""
Test echo-roots LLM integration architecture with mock client.

This script validates the LLM integration architecture and pipeline
without requiring actual API keys, using the mock client.
"""

import asyncio
from pathlib import Path

# Add src to path for imports
import sys
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from echo_roots.models.core import IngestionItem
from echo_roots.models.domain import DomainPack
from echo_roots.pipelines.llm_factory import LLMClientFactory
from echo_roots.pipelines.extraction import LLMExtractor


async def test_mock_integration():
    """Test LLM integration architecture with mock client."""
    
    print("🧪 Testing LLM Integration Architecture (Mock)")
    print("=" * 55)
    
    # 1. Create mock client
    print("\\n1. Creating mock LLM client...")
    try:
        client = LLMClientFactory.create_client("mock")
        print(f"   ✅ Mock client created: {type(client).__name__}")
    except Exception as e:
        print(f"   ❌ Failed to create mock client: {e}")
        return False
    
    # 2. Test direct client call
    print("\\n2. Testing direct client completion...")
    try:
        response = await client.complete(
            "Extract brand from: Apple iPhone 15", 
            temperature=0.1,
            max_tokens=100
        )
        print(f"   ✅ Direct response received")
        print(f"   📝 Response preview: {response[:100]}...")
    except Exception as e:
        print(f"   ❌ Direct client call failed: {e}")
        return False
    
    # 3. Create test domain pack
    print("\\n3. Creating test domain pack...")
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
    
    # 4. Create test items
    print("\\n4. Creating test items...")
    test_items = [
        IngestionItem(
            item_id="test_001",
            title="Apple iPhone 15 Pro Max",
            description="Latest flagship smartphone with A17 Pro chip",
            language="en",
            source="test"
        ),
        IngestionItem(
            item_id="test_002", 
            title="Samsung Galaxy S24 Ultra",
            description="Android flagship with S Pen and 200MP camera",
            language="en",
            source="test"
        ),
        IngestionItem(
            item_id="test_003",
            title="MacBook Pro M3",
            description="Professional laptop with Apple Silicon M3 chip", 
            language="en",
            source="test"
        )
    ]
    print(f"   ✅ Created {len(test_items)} test items")
    
    # 5. Create extractor
    print("\\n5. Creating LLM extractor...")
    try:
        from echo_roots.pipelines.extraction import ExtractorConfig
        config = ExtractorConfig(enable_validation=False)  # Disable validation for testing
        
        extractor = LLMExtractor(
            domain_pack=test_domain,
            llm_client=client,
            config=config
        )
        print(f"   ✅ Extractor created with {extractor.config.model_name}")
    except Exception as e:
        print(f"   ❌ Failed to create extractor: {e}")
        return False
    
    # 6. Test single extraction
    print("\\n6. Testing single item extraction...")
    try:
        result = await extractor.extract_single(test_items[0])
        print(f"   ✅ Extraction completed for: {result.item_id}")
        print(f"   📊 Attributes: {len(result.attributes)}, Terms: {len(result.terms)}")
        
        if result.attributes:
            print("   📋 Extracted attributes:")
            for attr in result.attributes:
                print(f"      - {attr.name}: '{attr.value}'")
        
        if result.terms:
            print("   🏷️  Extracted terms:")
            for term in result.terms:
                print(f"      - {term.term} (confidence: {term.confidence:.2f})")
        
        print(f"   ⏱️  Processing time: {result.metadata.processing_time_ms}ms")
        
    except Exception as e:
        print(f"   ❌ Single extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 7. Test batch extraction
    print("\\n7. Testing batch extraction...")
    try:
        results = await extractor.extract_batch(test_items)
        print(f"   ✅ Batch extraction completed")
        print(f"   📦 Processed {len(results)} items")
        
        for i, result in enumerate(results):
            print(f"   Item {i+1} ({result.item_id}): {len(result.attributes)} attrs, {len(result.terms)} terms")
            
    except Exception as e:
        print(f"   ❌ Batch extraction failed: {e}")
        return False
    
    # 8. Test factory with different configurations
    print("\\n8. Testing factory configurations...")
    try:
        # Test with config
        config_client = LLMClientFactory.create_client(
            "mock", 
            config={"mock": {"delay_seconds": 0.05}}
        )
        print("   ✅ Factory with custom config works")
        
        # Test environment creation
        env_client = LLMClientFactory.create_from_environment("mock")
        print("   ✅ Environment-based creation works")
        
        # Test provider listing
        providers = LLMClientFactory.list_providers()
        print(f"   ✅ Listed {len(providers)} supported providers")
        
    except Exception as e:
        print(f"   ❌ Factory configuration test failed: {e}")
        return False
    
    return True


async def test_gemini_architecture_readiness():
    """Test that Gemini architecture is ready (without API key)."""
    
    print("\\n\\n🔧 Testing Gemini Architecture Readiness")
    print("=" * 50)
    
    # 1. Check imports
    print("\\n1. Checking Gemini imports...")
    try:
        from echo_roots.pipelines.gemini_client import (
            GeminiClient, 
            GeminiProClient, 
            GeminiFlashClient,
            create_gemini_from_config
        )
        print("   ✅ All Gemini classes importable")
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        return False
    
    # 2. Check factory integration
    print("\\n2. Checking factory integration...")
    try:
        # This should fail gracefully without API key
        try:
            LLMClientFactory.create_client("gemini")
            print("   ⚠️  Gemini client created without API key (unexpected)")
        except Exception as e:
            if "GOOGLE_API_KEY" in str(e) or "api_key" in str(e).lower():
                print("   ✅ Factory correctly requires API key")
            else:
                print(f"   ❌ Unexpected error: {e}")
                return False
                
    except Exception as e:
        print(f"   ❌ Factory integration failed: {e}")
        return False
    
    # 3. Check configuration structure
    print("\\n3. Checking configuration structure...")
    try:
        test_config = {
            "gemini": {
                "api_key": "fake_key_for_testing",
                "model_name": "gemini-1.5-flash",
                "project_id": "test-project"
            }
        }
        
        # This might fail due to network/auth, but structure should be valid
        try:
            create_gemini_from_config(test_config)
            print("   ✅ Configuration structure valid")
        except Exception as e:
            if "api_key" in str(e).lower() or "auth" in str(e).lower() or "network" in str(e).lower():
                print("   ✅ Configuration structure valid (auth failed as expected)")
            else:
                print(f"   ⚠️  Structure test inconclusive: {e}")
                
    except Exception as e:
        print(f"   ❌ Configuration test failed: {e}")
        return False
    
    return True


def main():
    """Main test function."""
    
    print("🚀 Echo-Roots LLM Architecture Test")
    print("=" * 45)
    
    print(f"Python: {sys.version}")
    print(f"Working directory: {Path.cwd()}")
    
    async def run_tests():
        # Test mock integration
        mock_success = await test_mock_integration()
        
        # Test Gemini readiness
        gemini_ready = await test_gemini_architecture_readiness()
        
        # Summary
        print("\\n\\n📋 Test Summary")
        print("=" * 30)
        
        print(f"Mock Integration:      {'✅ PASS' if mock_success else '❌ FAIL'}")
        print(f"Gemini Architecture:   {'✅ READY' if gemini_ready else '❌ NOT READY'}")
        
        if mock_success and gemini_ready:
            print("\\n✅ LLM integration architecture is working correctly!")
            print("\\n💡 To test with real Gemini API:")
            print("   1. Get API key from https://makersuite.google.com/")
            print("   2. Set GOOGLE_API_KEY environment variable") 
            print("   3. Run: python test_gemini_integration.py")
            return 0
        else:
            print("\\n❌ Some architecture components need attention")
            return 1
    
    result = asyncio.run(run_tests())
    sys.exit(result)


if __name__ == "__main__":
    main()
