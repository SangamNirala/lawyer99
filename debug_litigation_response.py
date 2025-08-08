#!/usr/bin/env python3
"""
Debug Litigation Analytics Response Structure

Investigate the actual response structure to understand why some fields are empty
"""

import asyncio
import aiohttp
import json
import os

BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'https://713b7daa-6e2b-44d9-8b8d-1458f53c5728.preview.emergentagent.com')
API_BASE = f"{BACKEND_URL}/api"

async def debug_response_structure():
    """Debug the actual response structure"""
    
    case_data = {
        "case_type": "commercial",
        "jurisdiction": "federal",
        "case_facts": "Complex commercial dispute involving breach of contract.",
        "case_value": 1000000.0,
        "evidence_strength": 7.0
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{API_BASE}/litigation/analyze-case", json=case_data) as response:
            if response.status == 200:
                data = await response.json()
                print("📊 FULL RESPONSE STRUCTURE:")
                print("=" * 60)
                print(json.dumps(data, indent=2, default=str))
                print("=" * 60)
                
                print("\n🔍 FIELD ANALYSIS:")
                for key, value in data.items():
                    if isinstance(value, list):
                        print(f"  {key}: {len(value)} items - {value}")
                    elif isinstance(value, dict):
                        print(f"  {key}: dict with keys {list(value.keys())}")
                    else:
                        print(f"  {key}: {type(value).__name__} - {value}")
            else:
                print(f"❌ Error: HTTP {response.status}")
                print(await response.text())

if __name__ == "__main__":
    asyncio.run(debug_response_structure())