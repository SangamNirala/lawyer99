#!/usr/bin/env python3
"""
Test Legal Q&A Knowledge Base Initialization Endpoint
"""

import requests
import json
from datetime import datetime

def test_initialization():
    base_url = "https://9c72c586-5126-47dc-85cc-7afca9190d08.preview.emergentagent.com"
    api_url = f"{base_url}/api"
    
    print("🔍 Testing Knowledge Base Initialization...")
    print(f"URL: {api_url}/legal-qa/initialize-knowledge-base")
    print("⚠️  This may take up to 2 minutes...")
    
    try:
        response = requests.post(
            f"{api_url}/legal-qa/initialize-knowledge-base",
            json={},
            headers={'Content-Type': 'application/json'},
            timeout=120  # 2 minutes timeout
        )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ PASSED - Initialization successful")
            try:
                data = response.json()
                print(f"Response keys: {list(data.keys())}")
                if 'message' in data:
                    print(f"Message: {data['message']}")
                if 'rag_system_ready' in data:
                    print(f"RAG System Ready: {data['rag_system_ready']}")
                if 'knowledge_base_stats' in data:
                    stats = data['knowledge_base_stats']
                    if isinstance(stats, dict) and 'total_documents' in stats:
                        print(f"Total Documents: {stats['total_documents']}")
                return True
            except:
                print("Response received but not JSON")
                return True
        else:
            print(f"❌ FAILED - Status: {response.status_code}")
            try:
                error = response.json()
                print(f"Error: {error}")
            except:
                print(f"Error: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ FAILED - Request timed out after 2 minutes")
        return False
    except Exception as e:
        print(f"❌ FAILED - Exception: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_initialization()
    print(f"\nInitialization test: {'PASSED' if success else 'FAILED'}")