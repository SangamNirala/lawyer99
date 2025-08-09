import requests
import json
import random

def test_attorney_creation():
    """Test attorney creation with proper enum handling"""
    base_url = "https://9c72c586-5126-47dc-85cc-7afca9190d08.preview.emergentagent.com"
    api_url = f"{base_url}/api"
    
    attorney_data = {
        "email": f"priority_test_{random.randint(1000, 9999)}@legalmate.test",
        "first_name": "Priority",
        "last_name": "Tester",
        "bar_number": f"PRI{random.randint(100000, 999999)}",
        "jurisdiction": "California",
        "role": "supervising_attorney",
        "specializations": ["contract_law"],
        "years_experience": 5,
        "password": "PriorityTest123!"
    }
    
    print("🔍 Testing Attorney Creation...")
    response = requests.post(f"{api_url}/attorney/create", json=attorney_data, headers={'Content-Type': 'application/json'})
    print(f"   Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"   Response: {data}")
        if data.get('success'):
            print("   ✅ Attorney creation successful!")
            return data.get('attorney_id')
        else:
            print(f"   ❌ Attorney creation failed: {data.get('error')}")
            return None
    else:
        print(f"   ❌ HTTP Error: {response.text}")
        return None

def test_compliant_contract_generation():
    """Test compliant contract generation"""
    base_url = "https://9c72c586-5126-47dc-85cc-7afca9190d08.preview.emergentagent.com"
    api_url = f"{base_url}/api"
    
    contract_data = {
        "contract_type": "NDA",
        "jurisdiction": "US",
        "parties": {
            "party1_name": "Priority Test Corp",
            "party1_type": "corporation",
            "party2_name": "Contract Tester",
            "party2_type": "individual"
        },
        "terms": {
            "purpose": "Testing priority endpoint functionality",
            "duration": "1_year"
        },
        "special_clauses": []
    }
    
    print("\n🔍 Testing Compliant Contract Generation...")
    response = requests.post(f"{api_url}/generate-contract-compliant", json=contract_data, headers={'Content-Type': 'application/json'}, timeout=60)
    print(f"   Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"   Response keys: {list(data.keys())}")
        if 'contract' in data:
            contract = data['contract']
            print(f"   Contract ID: {contract.get('id')}")
            print(f"   Compliance Score: {contract.get('compliance_score')}%")
            print("   ✅ Compliant contract generation successful!")
            return True
        else:
            print(f"   ❌ No contract in response")
            return False
    else:
        print(f"   ❌ HTTP Error: {response.text}")
        return False

if __name__ == "__main__":
    print("🎯 PRIORITY ENDPOINTS TEST")
    print("=" * 50)
    
    # Test attorney creation
    attorney_id = test_attorney_creation()
    
    # Test compliant contract generation
    contract_success = test_compliant_contract_generation()
    
    print("\n" + "=" * 50)
    print("🎯 PRIORITY TEST RESULTS:")
    print(f"   Attorney Creation: {'✅ PASS' if attorney_id else '❌ FAIL'}")
    print(f"   Compliant Contract Generation: {'✅ PASS' if contract_success else '❌ FAIL'}")
    
    if attorney_id and contract_success:
        print("🎉 Both priority endpoints working!")
    else:
        print("⚠️  Some priority endpoints still need fixes")