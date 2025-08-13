#!/usr/bin/env python3
import requests
import json

def test_contract_generation():
    """Test contract generation with the fixed Groq model"""
    
    base_url = "https://legalmate-research.preview.emergentagent.com"
    api_url = f"{base_url}/api"
    
    # Test data for NDA generation
    nda_data = {
        "contract_type": "NDA",
        "jurisdiction": "US",
        "parties": {
            "party1_name": "TechCorp Solutions Inc.",
            "party1_type": "company",
            "party2_name": "Sarah Johnson",
            "party2_type": "individual"
        },
        "terms": {
            "purpose": "Evaluation of potential software development partnership and sharing of proprietary algorithms",
            "duration": "3_years"
        },
        "special_clauses": ["Return of materials clause", "Non-solicitation of employees"]
    }
    
    print("🔍 Testing Contract Generation with Fixed Groq Model...")
    print(f"   URL: {api_url}/generate-contract")
    
    try:
        response = requests.post(
            f"{api_url}/generate-contract",
            json=nda_data,
            headers={'Content-Type': 'application/json'},
            timeout=60
        )
        
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            contract = data.get('contract', {})
            
            print("✅ SUCCESS - Contract Generated!")
            print(f"   Contract ID: {contract.get('id')}")
            print(f"   Contract Type: {contract.get('contract_type')}")
            print(f"   Jurisdiction: {contract.get('jurisdiction')}")
            print(f"   Compliance Score: {contract.get('compliance_score')}%")
            print(f"   Content Length: {len(contract.get('content', ''))} characters")
            print(f"   Number of Clauses: {len(contract.get('clauses', []))}")
            
            # Check warnings and suggestions
            warnings = data.get('warnings', [])
            suggestions = data.get('suggestions', [])
            
            print(f"   Warnings: {len(warnings)} - {warnings}")
            print(f"   Suggestions: {len(suggestions)} - {suggestions}")
            
            # Show a snippet of the generated content
            content = contract.get('content', '')
            if content:
                print(f"\n   Content Preview (first 300 chars):")
                print(f"   {content[:300]}...")
            
            return True, data
        else:
            print(f"❌ FAILED - Status: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data}")
            except:
                print(f"   Error: {response.text}")
            return False, {}
            
    except Exception as e:
        print(f"❌ FAILED - Exception: {str(e)}")
        return False, {}

if __name__ == "__main__":
    print("🚀 Testing Groq Model Fix")
    print("=" * 50)
    
    success, result = test_contract_generation()
    
    if success:
        print("\n🎉 Groq model fix successful! Contract generation is working.")
    else:
        print("\n❌ Groq model fix failed. Contract generation still not working.")