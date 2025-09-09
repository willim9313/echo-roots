"""
Test Gemini API integration with echo-roots extraction pipeline.

This script validates that the Gemini client works properly with the
extraction pipeline using the official google.genai SDK.
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
from echo_roots.pipelines.extraction import ExtractionPipeline


async def test_gemini_integration():
    """Test Gemini client integration with extraction pipeline."""
    
    print("🔍 Testing Gemini LLM Integration")
    print("=" * 50)
    
    # 1. Check provider availability
    print("\\n1. Checking provider availability...")
    providers = get_available_providers()
    
    for provider, (available, message) in providers.items():
        status = "✅" if available else "❌"
        print(f"   {status} {provider}: {message}")
    
    # 2. Check if Gemini is available
    gemini_available, gemini_status = providers.get("gemini", (False, "Not found"))
    
    if not gemini_available:
        print(f"\\n❌ Gemini not available: {gemini_status}")
        if "GOOGLE_API_KEY" in gemini_status:
            print("\\n💡 To test Gemini integration:")
            print("   1. Get API key from https://makersuite.google.com/")
            print("   2. Set environment variable: export GOOGLE_API_KEY=your_key_here")
            print("   3. Run this test again")
        return False
    
    print(f"\\n✅ Gemini is available: {gemini_status}")
    
    # 3. Create test data
    print("\\n2. Creating test data...")
    test_item = IngestionItem(
        item_id="test_001",
        title="Apple iPhone 15 Pro Max",
        description="Latest flagship smartphone from Apple with advanced camera system, A17 Pro chip, and titanium design. Available in multiple colors with premium build quality.",
        language="en",
        source="test_data"
    )
    print(f"   📱 Test item: {test_item.title}")
    
    # 4. Create Gemini client
    print("\\n3. Creating Gemini client...")
    try:
        client = LLMClientFactory.create_client("gemini")
        print(f"   ✅ Gemini client created: {type(client).__name__}")
        
        # Test direct client call
        print("\\n4. Testing direct client call...")
        test_prompt = "Extract the brand name from this text: Apple iPhone 15 Pro Max"
        response = await client.complete(test_prompt, max_tokens=100)
        print(f"   📝 Direct response: {response[:100]}...")
        
    except Exception as e:
        print(f"   ❌ Failed to create Gemini client: {e}")
        return False
    
    # 5. Test with extraction pipeline (mock domain)
    print("\\n5. Testing with extraction pipeline...")
    try:
        # Create a simple test domain configuration
        from echo_roots.models.domain import DomainPack
        
        test_domain = DomainPack(
            domain="electronics",
            taxonomy_version="1.0",
            output_schema={
                "attributes": [
                    {"key": "brand", "type": "string"},
                    {"key": "category", "type": "string"},
                    {"key": "model", "type": "string"}
                ]
            },
            attribute_hints={
                "brand": {
                    "examples": ["Apple", "Samsung", "Google"],
                    "description": "Manufacturer or brand name"
                },
                "category": {
                    "examples": ["smartphone", "laptop", "tablet"],
                    "description": "Product category"
                }
            }
        )
        
        # Test extraction
        from echo_roots.pipelines.extraction import LLMExtractor
        
        extractor = LLMExtractor(
            domain_pack=test_domain,
            llm_client=client
        )
        
        result = await extractor.extract_single(test_item)
        
        print(f"   ✅ Extraction completed for item: {result.item_id}")
        print(f"   📊 Extracted {len(result.attributes)} attributes and {len(result.terms)} terms")
        
        # Display results
        if result.attributes:
            print("\\n   📋 Extracted Attributes:")
            for attr in result.attributes:
                print(f"      - {attr.name}: {attr.value}")
                if attr.evidence:
                    print(f"        Evidence: {attr.evidence[:80]}...")
        
        if result.terms:
            print("\\n   🏷️  Extracted Terms:")
            for term in result.terms:
                print(f"      - {term.term} (confidence: {term.confidence:.2f})")
                if term.context:
                    print(f"        Context: {term.context[:60]}...")
        
        print(f"\\n   ⏱️  Processing time: {result.metadata.processing_time_ms}ms")
        print(f"   🤖 Model used: {result.metadata.model}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Extraction pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_multiple_providers():
    """Test multiple LLM providers if available."""
    
    print("\\n\\n🔄 Testing Multiple Provider Support")
    print("=" * 50)
    
    providers_to_test = []
    providers = get_available_providers()
    
    for provider, (available, _) in providers.items():
        if available and provider != "mock":
            providers_to_test.append(provider)
    
    if not providers_to_test:
        print("❌ No real providers available for testing")
        return
    
    print(f"✅ Testing providers: {', '.join(providers_to_test)}")
    
    # Create simple test prompt
    test_prompt = "What is the capital of France? Answer in one word."
    
    for provider in providers_to_test:
        try:
            print(f"\\n🧪 Testing {provider}...")
            client = LLMClientFactory.create_client(provider)
            
            start_time = asyncio.get_event_loop().time()
            response = await client.complete(test_prompt, max_tokens=50)
            end_time = asyncio.get_event_loop().time()
            
            print(f"   ✅ Response: {response.strip()}")
            print(f"   ⏱️  Time: {(end_time - start_time)*1000:.0f}ms")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")


def main():
    """Main test function."""
    
    print("🚀 Echo-Roots Gemini Integration Test")
    print("=" * 60)
    
    # Check Python environment
    print(f"Python: {sys.version}")
    print(f"Working directory: {Path.cwd()}")
    
    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        print(f"Google API Key: {'*' * (len(api_key) - 8) + api_key[-4:]}")
    else:
        print("Google API Key: Not set")
    
    # Run async tests
    async def run_tests():
        success = await test_gemini_integration()
        
        if success:
            await test_multiple_providers()
            print("\\n\\n✅ All tests completed successfully!")
            print("\\n💡 Gemini integration is working properly with echo-roots")
        else:
            print("\\n\\n❌ Some tests failed. Check configuration and try again.")
            return 1
        
        return 0
    
    # Run the tests
    result = asyncio.run(run_tests())
    sys.exit(result)


if __name__ == "__main__":
    main()
