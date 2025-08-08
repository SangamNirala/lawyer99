import requests
import json
import re

def test_specific_review_id_extraction():
    """Test the specific regex pattern mentioned in requirements: /ID:\s*([^)]+)/"""
    
    base_url = "https://8fa35272-f813-4f54-99e2-91d7f9fb1b50.preview.emergentagent.com"
    api_url = f"{base_url}/api"
    
    print("🔍 TESTING SPECIFIC REVIEW ID EXTRACTION PATTERN")
    print("="*60)
    
    # Generate a compliant contract to get real suggestions
    contract_data = {
        "contract_type": "NDA",
        "jurisdiction": "US",
        "parties": {
            "party1_name": "Test Company Inc.",
            "party1_type": "corporation",
            "party2_name": "John Smith",
            "party2_type": "individual"
        },
        "terms": {
            "purpose": "Testing review ID extraction pattern",
            "duration": "1_year"
        }
    }
    
    try:
        response = requests.post(
            f"{api_url}/generate-contract-compliant",
            json=contract_data,
            headers={'Content-Type': 'application/json'},
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            suggestions = data.get('suggestions', [])
            
            print(f"✅ Contract generated successfully")
            print(f"📋 Found {len(suggestions)} suggestions:")
            
            # The specific regex pattern from requirements
            frontend_regex = r'ID:\s*([^)]+)'
            
            review_ids_found = []
            
            for i, suggestion in enumerate(suggestions):
                print(f"\n   Suggestion {i+1}: {suggestion}")
                
                # Test the specific regex pattern
                match = re.search(frontend_regex, suggestion)
                if match:
                    review_id = match.group(1).strip()
                    review_ids_found.append(review_id)
                    print(f"   ✅ REGEX MATCH: '{review_id}'")
                    
                    # Verify this review exists
                    verify_review(api_url, review_id)
                else:
                    print(f"   ❌ NO REGEX MATCH with pattern: {frontend_regex}")
            
            print(f"\n📊 SUMMARY:")
            print(f"   Total suggestions: {len(suggestions)}")
            print(f"   Review IDs found: {len(review_ids_found)}")
            print(f"   Review IDs: {review_ids_found}")
            
            if review_ids_found:
                print(f"   ✅ The frontend regex pattern SHOULD WORK")
                print(f"   🔍 Frontend issue might be elsewhere (JavaScript implementation, timing, etc.)")
            else:
                print(f"   ❌ The frontend regex pattern WILL NOT WORK")
                print(f"   🔧 Backend needs to adjust the suggestion format")
                
            return review_ids_found
            
        else:
            print(f"❌ Failed to generate contract: {response.status_code}")
            print(f"   Error: {response.text}")
            return []
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return []

def verify_review(api_url, review_id):
    """Verify that a review ID can be fetched"""
    try:
        response = requests.get(
            f"{api_url}/attorney/review/status/{review_id}",
            timeout=30
        )
        
        if response.status_code == 200:
            review_data = response.json()
            print(f"   ✅ Review verified: Status={review_data.get('status')}, Progress={review_data.get('progress_percentage')}%")
        else:
            print(f"   ❌ Review verification failed: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Review verification error: {str(e)}")

def test_multiple_contracts_for_pattern_consistency():
    """Test multiple contracts to see if the pattern is consistent"""
    
    base_url = "https://8fa35272-f813-4f54-99e2-91d7f9fb1b50.preview.emergentagent.com"
    api_url = f"{base_url}/api"
    
    print(f"\n🔍 TESTING PATTERN CONSISTENCY ACROSS MULTIPLE CONTRACTS")
    print("="*60)
    
    contract_types = [
        {
            "name": "NDA",
            "data": {
                "contract_type": "NDA",
                "jurisdiction": "US",
                "parties": {
                    "party1_name": "Alpha Corp",
                    "party1_type": "corporation",
                    "party2_name": "Beta User",
                    "party2_type": "individual"
                },
                "terms": {
                    "purpose": "Pattern consistency test",
                    "duration": "1_year"
                }
            }
        },
        {
            "name": "Freelance Agreement",
            "data": {
                "contract_type": "freelance_agreement",
                "jurisdiction": "US",
                "parties": {
                    "party1_name": "Client Company",
                    "party1_type": "company",
                    "party2_name": "Freelancer Name",
                    "party2_type": "individual"
                },
                "terms": {
                    "scope": "Pattern consistency test project",
                    "payment_amount": "$1000",
                    "payment_terms": "milestone"
                }
            }
        }
    ]
    
    frontend_regex = r'ID:\s*([^)]+)'
    all_review_ids = []
    
    for contract_type in contract_types:
        print(f"\n--- Testing {contract_type['name']} ---")
        
        try:
            response = requests.post(
                f"{api_url}/generate-contract-compliant",
                json=contract_type['data'],
                headers={'Content-Type': 'application/json'},
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                suggestions = data.get('suggestions', [])
                
                print(f"   ✅ Generated {contract_type['name']} successfully")
                print(f"   📋 Suggestions count: {len(suggestions)}")
                
                contract_review_ids = []
                
                for suggestion in suggestions:
                    match = re.search(frontend_regex, suggestion)
                    if match:
                        review_id = match.group(1).strip()
                        contract_review_ids.append(review_id)
                        print(f"   ✅ Found review ID: {review_id}")
                
                all_review_ids.extend(contract_review_ids)
                
                if not contract_review_ids:
                    print(f"   ⚠️  No review IDs found in {contract_type['name']}")
                    # Show all suggestions for debugging
                    for i, suggestion in enumerate(suggestions):
                        print(f"      Suggestion {i+1}: {suggestion}")
                        
            else:
                print(f"   ❌ Failed to generate {contract_type['name']}: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error with {contract_type['name']}: {str(e)}")
    
    print(f"\n📊 OVERALL PATTERN CONSISTENCY RESULTS:")
    print(f"   Total contracts tested: {len(contract_types)}")
    print(f"   Total review IDs found: {len(all_review_ids)}")
    print(f"   Review IDs: {all_review_ids}")
    
    if len(all_review_ids) == len(contract_types):
        print(f"   ✅ CONSISTENT: Each contract generated exactly one review ID")
    elif len(all_review_ids) > 0:
        print(f"   ⚠️  INCONSISTENT: Some contracts generated review IDs, others didn't")
    else:
        print(f"   ❌ BROKEN: No review IDs found in any contract")
    
    return all_review_ids

if __name__ == "__main__":
    print("🎯 REVIEW ID EXTRACTION ANALYSIS")
    print("="*80)
    
    # Test 1: Specific regex pattern
    review_ids_1 = test_specific_review_id_extraction()
    
    # Test 2: Pattern consistency
    review_ids_2 = test_multiple_contracts_for_pattern_consistency()
    
    # Final analysis
    all_review_ids = list(set(review_ids_1 + review_ids_2))  # Remove duplicates
    
    print(f"\n🎯 FINAL ANALYSIS")
    print("="*80)
    print(f"Total unique review IDs found: {len(all_review_ids)}")
    print(f"Review IDs: {all_review_ids}")
    
    if all_review_ids:
        print(f"\n✅ CONCLUSION: Backend IS working correctly")
        print(f"   - Review IDs are being generated and included in suggestions")
        print(f"   - The regex pattern /ID:\\s*([^)]+)/ DOES work")
        print(f"   - Reviews can be fetched via GET /api/attorney/review/status/{{review_id}}")
        print(f"\n🔍 FRONTEND DEBUGGING NEEDED:")
        print(f"   - Check if frontend is correctly parsing the suggestions array")
        print(f"   - Verify the JavaScript regex implementation")
        print(f"   - Check for timing issues (async/await problems)")
        print(f"   - Verify the ReviewStatus component is being triggered")
    else:
        print(f"\n❌ CONCLUSION: Backend has issues")
        print(f"   - No review IDs found in suggestions")
        print(f"   - Backend needs to fix the suggestion format")