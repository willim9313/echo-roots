#!/usr/bin/env python3
"""
LLM 提取測試腳本
Test script for LLM extraction functionality
"""

import asyncio
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from echo_roots.models.core import IngestionItem, ExtractionResult, AttributeExtraction, SemanticTerm, ExtractionMetadata
from echo_roots.models.domain import DomainPack
from echo_roots.pipelines.extraction import LLMExtractor, ExtractorConfig
from echo_roots.storage.duckdb_backend import DuckDBStorageManager


async def load_test_data():
    """載入測試資料"""
    print("📄 Loading test data from storage...")
    
    # 配置儲存
    storage_config = {
        "duckdb": {
            "database_path": "./test_echo_roots.db"
        }
    }
    
    storage_manager = DuckDBStorageManager(storage_config)
    await storage_manager.initialize()
    
    # 獲取儲存的測試資料
    ingestion_repo = storage_manager.ingestion
    items = await ingestion_repo.list_items(limit=5)
    
    print(f"✅ Loaded {len(items)} items from storage")
    for i, item in enumerate(items, 1):
        print(f"  {i}. {item.title} ({item.raw_category})")
    
    await storage_manager.close()
    return items


async def test_mock_llm_extraction(items: List[IngestionItem]):
    """測試模擬 LLM 提取"""
    print("\n🤖 Testing Mock LLM Extraction...")
    
    # 載入領域包配置
    domain_path = Path("domains/ecommerce")
    domain_file = domain_path / "domain.yaml"
    
    with open(domain_file, 'r', encoding='utf-8') as f:
        domain_config = yaml.safe_load(f)
    
    domain_pack = DomainPack(**domain_config)
    
    # 簡化的模擬提取器
    async def mock_extract_attributes(item: IngestionItem) -> ExtractionResult:
        """模擬屬性提取"""
        
        # 根據商品類型模擬提取不同屬性
        mock_attributes = []
        mock_terms = []
        
        # 分析標題和描述，模擬 LLM 理解
        title_lower = item.title.lower()
        description_lower = (item.description or "").lower()
        
        # 品牌提取
        brand_keywords = {
            'apple': 'Apple',
            'iphone': 'Apple', 
            'samsung': 'Samsung',
            'galaxy': 'Samsung',
            'lenovo': 'Lenovo',
            'thinkpad': 'Lenovo',
            'sony': 'Sony',
            'nintendo': 'Nintendo',
            'asus': 'ASUS',
            'xiaomi': 'Xiaomi',
            '小米': 'Xiaomi'
        }
        
        for keyword, brand in brand_keywords.items():
            if keyword in title_lower or keyword in description_lower:
                mock_attributes.append(AttributeExtraction(
                    name="brand",
                    value=brand,
                    evidence=f"Found '{keyword}' in title",
                    confidence=0.95
                ))
                break
        
        # 顏色提取
        color_keywords = {
            '黑色': 'black', '白色': 'white', '銀色': 'silver',
            '金色': 'gold', '藍色': 'blue', '紅色': 'red',
            '灰色': 'gray', '鈦金屬': 'titanium', '天然鈦金屬': 'natural titanium',
            '鈦灰色': 'titanium gray'
        }
        
        for color_cn, color_en in color_keywords.items():
            if color_cn in item.title or color_cn in (item.description or ""):
                mock_attributes.append(AttributeExtraction(
                    name="color",
                    value=color_en,
                    evidence=f"Found color '{color_cn}' in text",
                    confidence=0.90
                ))
                break
        
        # 容量/尺寸提取
        import re
        
        # 儲存容量
        storage_match = re.search(r'(\d+)GB', item.title)
        if storage_match:
            capacity = storage_match.group(1) + "GB"
            mock_attributes.append(AttributeExtraction(
                name="storage_capacity",
                value=capacity,
                evidence=f"Extracted storage: {capacity}",
                confidence=0.98
            ))
        
        # 螢幕尺寸
        screen_match = re.search(r'(\d+\.?\d*)吋', item.title + " " + (item.description or ""))
        if screen_match:
            size = screen_match.group(1) + "吋"
            mock_attributes.append(AttributeExtraction(
                name="screen_size",
                value=size,
                evidence=f"Extracted screen size: {size}",
                confidence=0.90
            ))
        
        # 價格範圍（基於實際價格）
        if 'price' in item.raw_attributes:
            price = item.raw_attributes['price']
            if price < 1000:
                price_range = "budget"
            elif price < 10000:
                price_range = "mid-range"
            else:
                price_range = "premium"
            
            mock_attributes.append(AttributeExtraction(
                name="price_range",
                value=price_range,
                evidence=f"Price {price} categorized as {price_range}",
                confidence=0.85
            ))
        
        # 術語提取（關鍵詞）
        tech_terms = [
            'A17 Pro', 'AMOLED', 'OLED', 'Retina', 'Dynamic', 
            '主動式降噪', '藍牙', 'WiFi', 'USB-C', '快充',
            '防水', '指紋', 'Face ID', 'NFC'
        ]
        
        text_content = item.title + " " + (item.description or "")
        for term in tech_terms:
            if term in text_content:
                mock_terms.append(SemanticTerm(
                    term=term,
                    context=f"Found in product description",
                    confidence=0.80
                ))
        
        # 創建提取結果
        result = ExtractionResult(
            item_id=item.item_id,
            attributes=mock_attributes,
            terms=mock_terms,
            metadata=ExtractionMetadata(
                model="mock-llm-v1",
                run_id=f"mock-run-{hash(item.item_id)}",
                extracted_at="2025-09-08T22:45:00Z",
                processing_time_ms=150  # 模擬處理時間
            )
        )
        
        return result
    
    # 測試提取
    for i, item in enumerate(items[:3], 1):
        print(f"\n🔍 Processing item {i}: {item.title}")
        
        try:
            result = await mock_extract_attributes(item)
            
            print(f"✅ Extracted {len(result.attributes)} attributes:")
            for attr in result.attributes:
                print(f"  - {attr.name}: {attr.value} (confidence: {attr.confidence:.2f})")
                print(f"    Evidence: {attr.evidence}")
            
            print(f"✅ Extracted {len(result.terms)} terms:")
            for term in result.terms[:3]:  # 只顯示前3個
                print(f"  - {term.term} (confidence: {term.confidence:.2f})")
            
            print(f"⏱️ Processing time: {result.metadata.processing_time_ms}ms")
            
        except Exception as e:
            print(f"❌ Failed to extract from item {i}: {str(e)}")


async def test_category_mapping(items: List[IngestionItem]):
    """測試分類映射"""
    print("\n🏷️ Testing Category Mapping...")
    
    # 載入分類結構
    taxonomy_path = Path("domains/ecommerce/categories/taxonomy.yaml")
    with open(taxonomy_path, 'r', encoding='utf-8') as f:
        taxonomy_config = yaml.safe_load(f)
    
    categories = taxonomy_config['categories']
    
    # 建立分類映射字典
    category_map = {}
    
    def build_category_paths(cats, parent_path=""):
        for cat in cats:
            current_path = f"{parent_path}/{cat['name']}" if parent_path else cat['name']
            category_map[cat['name']] = {
                'id': cat['id'],
                'path': current_path,
                'level': cat['level']
            }
            
            if 'children' in cat:
                build_category_paths(cat['children'], current_path)
    
    build_category_paths(categories)
    
    print(f"📚 Loaded {len(category_map)} categories from taxonomy")
    
    # 測試映射
    mapping_results = []
    
    for item in items[:5]:
        print(f"\n📍 Mapping: {item.title}")
        print(f"   Raw category: {item.raw_category}")
        
        # 簡單的關鍵詞匹配
        best_match = None
        best_score = 0
        
        if item.raw_category:
            raw_parts = item.raw_category.replace('>', '/').split('/')
            
            for cat_name, cat_info in category_map.items():
                # 計算匹配分數
                score = 0
                for part in raw_parts:
                    if part.strip() in cat_name or cat_name in part.strip():
                        score += 1
                
                if score > best_score:
                    best_score = score
                    best_match = cat_info
        
        if best_match:
            print(f"   ✅ Mapped to: {best_match['path']} (score: {best_score})")
            mapping_results.append({
                'item_id': item.item_id,
                'raw_category': item.raw_category,
                'mapped_category': best_match['path'],
                'confidence': best_score / len(item.raw_category.split('/')) if item.raw_category else 0
            })
        else:
            print(f"   ❌ No mapping found")
            mapping_results.append({
                'item_id': item.item_id,
                'raw_category': item.raw_category,
                'mapped_category': None,
                'confidence': 0
            })
    
    # 顯示映射統計
    mapped_count = sum(1 for r in mapping_results if r['mapped_category'])
    avg_confidence = sum(r['confidence'] for r in mapping_results) / len(mapping_results)
    
    print(f"\n📊 Mapping Statistics:")
    print(f"   Mapped: {mapped_count}/{len(mapping_results)} ({mapped_count/len(mapping_results)*100:.1f}%)")
    print(f"   Average confidence: {avg_confidence:.2f}")


async def test_data_quality_analysis(items: List[IngestionItem]):
    """測試資料品質分析"""
    print("\n📊 Testing Data Quality Analysis...")
    
    quality_metrics = {
        'total_items': len(items),
        'with_description': 0,
        'with_category': 0,
        'with_attributes': 0,
        'avg_title_length': 0,
        'avg_description_length': 0,
        'unique_sources': set(),
        'languages': set(),
        'attribute_coverage': {}
    }
    
    total_title_length = 0
    total_desc_length = 0
    
    for item in items:
        # 基本完整性
        if item.description:
            quality_metrics['with_description'] += 1
            total_desc_length += len(item.description)
        
        if item.raw_category:
            quality_metrics['with_category'] += 1
        
        if item.raw_attributes:
            quality_metrics['with_attributes'] += 1
        
        total_title_length += len(item.title)
        
        # 來源和語言
        quality_metrics['unique_sources'].add(item.source)
        quality_metrics['languages'].add(item.language)
        
        # 屬性覆蓋率
        for attr_key in item.raw_attributes.keys():
            if attr_key not in quality_metrics['attribute_coverage']:
                quality_metrics['attribute_coverage'][attr_key] = 0
            quality_metrics['attribute_coverage'][attr_key] += 1
    
    # 計算平均值
    quality_metrics['avg_title_length'] = total_title_length / len(items)
    quality_metrics['avg_description_length'] = total_desc_length / quality_metrics['with_description'] if quality_metrics['with_description'] > 0 else 0
    
    # 顯示結果
    print(f"📈 Data Quality Metrics:")
    print(f"   Total items: {quality_metrics['total_items']}")
    print(f"   With description: {quality_metrics['with_description']} ({quality_metrics['with_description']/len(items)*100:.1f}%)")
    print(f"   With category: {quality_metrics['with_category']} ({quality_metrics['with_category']/len(items)*100:.1f}%)")
    print(f"   With attributes: {quality_metrics['with_attributes']} ({quality_metrics['with_attributes']/len(items)*100:.1f}%)")
    print(f"   Avg title length: {quality_metrics['avg_title_length']:.1f} chars")
    print(f"   Avg description length: {quality_metrics['avg_description_length']:.1f} chars")
    print(f"   Unique sources: {len(quality_metrics['unique_sources'])}")
    print(f"   Languages: {list(quality_metrics['languages'])}")
    
    print(f"\n🏷️ Top attributes:")
    sorted_attrs = sorted(quality_metrics['attribute_coverage'].items(), key=lambda x: x[1], reverse=True)
    for attr, count in sorted_attrs[:8]:
        coverage = count / len(items) * 100
        print(f"   {attr}: {count} items ({coverage:.1f}%)")


async def main():
    """主要測試流程"""
    print("🤖 Echo-Roots LLM Extraction Test")
    print("=" * 50)
    
    try:
        # 1. 載入測試資料
        items = await load_test_data()
        if not items:
            print("❌ No test data found. Please run test_data_processing.py first.")
            return
        
        # 2. 測試模擬 LLM 提取
        await test_mock_llm_extraction(items)
        
        # 3. 測試分類映射
        await test_category_mapping(items)
        
        # 4. 測試資料品質分析
        await test_data_quality_analysis(items)
        
        print("\n🎉 LLM extraction tests completed successfully!")
        print("\nNext steps:")
        print("1. Integrate real LLM provider (OpenAI, Anthropic, etc.)")
        print("2. Fine-tune extraction prompts")
        print("3. Test semantic enrichment pipeline")
        print("4. Validate taxonomy mapping accuracy")
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
