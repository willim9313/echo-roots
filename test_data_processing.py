#!/usr/bin/env python3
"""
測試資料處理腳本
Test script for processing the prepared data
"""

import asyncio
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from echo_roots.models.core import IngestionItem
from echo_roots.models.domain import DomainPack
from echo_roots.domain.adapter import DomainAdapter
from echo_roots.pipelines.ingestion import IngestionPipeline, IngestionConfig
from echo_roots.storage.duckdb_backend import DuckDBStorageManager


async def load_domain_pack(domain_path: Path) -> DomainPack:
    """載入領域包配置"""
    print(f"📦 Loading domain pack from: {domain_path}")
    
    # 讀取主要配置
    domain_file = domain_path / "domain.yaml"
    with open(domain_file, 'r', encoding='utf-8') as f:
        domain_config = yaml.safe_load(f)
    
    # 讀取分類結構
    taxonomy_file = domain_path / "categories" / "taxonomy.yaml"
    taxonomy_config = None
    if taxonomy_file.exists():
        with open(taxonomy_file, 'r', encoding='utf-8') as f:
            taxonomy_config = yaml.safe_load(f)
    
    print(f"✅ Domain: {domain_config['domain']}")
    print(f"✅ Version: {domain_config['taxonomy_version']}")
    if taxonomy_config:
        print(f"✅ Categories: {len(taxonomy_config['categories'])} top-level")
    
    return DomainPack(**domain_config)


async def load_sample_data(data_path: Path) -> List[Dict[str, Any]]:
    """載入範例商品資料"""
    print(f"📄 Loading sample data from: {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✅ Loaded {len(data)} sample products")
    
    # 顯示前幾筆資料的摘要
    for i, item in enumerate(data[:3]):
        print(f"  {i+1}. {item['title']} ({item['raw_category']})")
    
    return data


async def test_domain_adapter(domain_pack: DomainPack, sample_data: List[Dict[str, Any]]):
    """測試領域適配器"""
    print("\n🔄 Testing Domain Adapter...")
    
    adapter = DomainAdapter(domain_pack)
    
    # 測試單筆轉換
    first_item = sample_data[0]
    
    # 手動創建 IngestionItem 來測試基本功能
    print(f"📝 Manually creating IngestionItem for: {first_item['title']}")
    
    ingestion_item = IngestionItem(
        item_id=first_item.get('product_id', f"test-{hash(first_item['title'])}"),
        title=first_item['title'],
        description=first_item.get('description'),
        raw_category=first_item.get('raw_category'),
        raw_attributes={
            k: v for k, v in first_item.items() 
            if k not in ['title', 'description', 'raw_category', 'product_id']
        },
        source="test_data",
        language=first_item.get('language', 'zh-TW')
    )
    
    print(f"✅ Successfully created: {ingestion_item.title}")
    print(f"  - ID: {ingestion_item.item_id}")
    print(f"  - Language: {ingestion_item.language}")
    print(f"  - Raw Category: {ingestion_item.raw_category}")
    print(f"  - Attributes count: {len(ingestion_item.raw_attributes)}")
    
    # 創建更多測試項目
    test_items = []
    for i, item in enumerate(sample_data[:5]):
        test_item = IngestionItem(
            item_id=item.get('product_id', f"test-{i}"),
            title=item['title'],
            description=item.get('description'),
            raw_category=item.get('raw_category'),
            raw_attributes={
                k: v for k, v in item.items() 
                if k not in ['title', 'description', 'raw_category', 'product_id']
            },
            source="test_batch",
            language=item.get('language', 'zh-TW')
        )
        test_items.append(test_item)
    
    print(f"✅ Created {len(test_items)} test items")
    
    return test_items


async def test_storage_integration(adapted_items: List[IngestionItem]):
    """測試儲存整合"""
    print("\n💾 Testing Storage Integration...")
    
    # 初始化儲存配置
    storage_config = {
        "duckdb": {
            "database_path": "./test_echo_roots.db"
        }
    }
    
    storage_manager = DuckDBStorageManager(storage_config)
    await storage_manager.initialize()
    
    # 獲取攝取儲存庫
    ingestion_repo = storage_manager.ingestion
    
    # 儲存測試資料
    stored_ids = []
    for item in adapted_items[:3]:  # 只測試前3筆
        try:
            item_id = await ingestion_repo.store_item(item)
            stored_ids.append(item_id)
            print(f"✅ Stored item: {item_id}")
        except Exception as e:
            print(f"❌ Failed to store item {item.item_id}: {str(e)}")
    
    # 測試查詢
    print(f"\n🔍 Testing retrieval...")
    for item_id in stored_ids[:2]:
        retrieved = await ingestion_repo.get_item(item_id)
        if retrieved:
            print(f"✅ Retrieved: {retrieved.title}")
        else:
            print(f"❌ Failed to retrieve: {item_id}")
    
    # 列出所有項目
    all_items = await ingestion_repo.list_items(limit=10)
    print(f"📋 Total items in storage: {len(all_items)}")
    
    await storage_manager.close()
    return stored_ids


async def test_category_matching(domain_pack: DomainPack, sample_data: List[Dict[str, Any]]):
    """測試分類匹配"""
    print("\n🏷️ Testing Category Matching...")
    
    # 收集所有原始分類
    raw_categories = set()
    for item in sample_data:
        if 'raw_category' in item:
            raw_categories.add(item['raw_category'])
    
    print(f"Found {len(raw_categories)} unique raw categories:")
    
    # 分析分類格式
    category_patterns = {}
    for cat in list(raw_categories)[:10]:  # 只顯示前10個
        parts = cat.replace('>', '/').split('/')
        pattern = f"{len(parts)} levels"
        if pattern not in category_patterns:
            category_patterns[pattern] = []
        category_patterns[pattern].append(cat)
        print(f"  - {cat}")
    
    print(f"\nCategory patterns:")
    for pattern, examples in category_patterns.items():
        print(f"  {pattern}: {len(examples)} categories")
        if examples:
            print(f"    Example: {examples[0]}")


async def main():
    """主要測試流程"""
    print("🌱 Echo-Roots Data Processing Test")
    print("=" * 50)
    
    # 設定路徑
    root_path = Path(".")
    domain_path = root_path / "domains" / "ecommerce"
    data_path = root_path / "data" / "raw" / "products" / "sample_products.json"
    
    try:
        # 1. 載入領域包
        domain_pack = await load_domain_pack(domain_path)
        
        # 2. 載入範例資料
        sample_data = await load_sample_data(data_path)
        
        # 3. 測試領域適配器
        adapted_items = await test_domain_adapter(domain_pack, sample_data)
        if not adapted_items:
            print("❌ Domain adapter test failed")
            return
        
        # 4. 測試分類匹配分析
        await test_category_matching(domain_pack, sample_data)
        
        # 5. 測試儲存整合
        await test_storage_integration(adapted_items)
        
        print("\n🎉 All tests completed successfully!")
        print("\nNext steps:")
        print("1. Check the test database: test_echo_roots.db")
        print("2. Run LLM extraction on the stored data")
        print("3. Test taxonomy mapping and enrichment")
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
