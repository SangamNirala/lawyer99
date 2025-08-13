import requests
import json
import random

def debug_review_action():
    """Debug the attorney review action issue"""
    base_url = "https://legal-engine-check.preview.emergentagent.com"
    api_url = f"{base_url}/api"
    
    # Step 1: Create an attorney
    print("🔍 Step 1: Creating Attorney...")
    attorney_data = {
        "email": f"debug_attorney_{random.randint(1000, 9999)}@legalmate.test",
        "first_name": "Debug",
        "last_name": "Attorney",
        "bar_number": f"DBG{random.randint(100000, 999999)}",
        "jurisdiction": "California",
        "role": "supervising_attorney",
        "specializations": ["contract_law"],
        "years_experience": 10,
        "password": "DebugTest123!"
    }
    
    response = requests.post(f"{api_url}/attorney/create", json=attorney_data)
    if response.status_code != 200:
        print(f"   ❌ Failed to create attorney: {response.text}")
        return
    
    attorney_result = response.json()
    if not attorney_result.get('success'):
        print(f"   ❌ Attorney creation failed: {attorney_result}")
        return
    
    attorney_id = attorney_result.get('attorney_id')
    print(f"   ✅ Attorney created: {attorney_id}")
    
    # Step 2: Submit a document for review
    print("\n🔍 Step 2: Submitting Document for Review...")
    document_data = {
        "document_content": "Debug test contract content for attorney review action testing.",
        "document_type": "contract",
        "client_id": f"debug_client_{random.randint(1000, 9999)}",
        "original_request": {
            "contract_type": "NDA",
            "jurisdiction": "US"
        },
        "priority": "high"
    }
    
    response = requests.post(f"{api_url}/attorney/review/submit", json=document_data)
    if response.status_code != 200:
        print(f"   ❌ Failed to submit document: {response.text}")
        return
    
    review_result = response.json()
    if not review_result.get('success'):
        print(f"   ❌ Document submission failed: {review_result}")
        return
    
    review_id = review_result.get('review_id')
    print(f"   ✅ Document submitted: {review_id}")
    
    # Step 3: Check review status
    print(f"\n🔍 Step 3: Checking Review Status...")
    response = requests.get(f"{api_url}/attorney/review/status/{review_id}")
    if response.status_code != 200:
        print(f"   ❌ Failed to get review status: {response.text}")
        return
    
    status_result = response.json()
    print(f"   Review Status: {json.dumps(status_result, indent=2)}")
    
    assigned_attorney = status_result.get('attorney')
    print(f"   Assigned Attorney: {assigned_attorney}")
    print(f"   Our Attorney ID: {attorney_id}")
    
    # Step 4: Try review action with the assigned attorney (if different)
    print(f"\n🔍 Step 4: Attempting Review Action...")
    
    # Use the assigned attorney ID if available, otherwise use our created attorney
    action_attorney_id = assigned_attorney if assigned_attorney else attorney_id
    
    action_data = {
        "review_id": review_id,
        "attorney_id": action_attorney_id,
        "action": "approve",
        "comments": "Debug test approval",
        "approved_content": "Approved debug test contract content."
    }
    
    print(f"   Using attorney ID: {action_attorney_id}")
    response = requests.post(f"{api_url}/attorney/review/action", json=action_data)
    print(f"   Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"   ✅ Review action successful: {result}")
    else:
        print(f"   ❌ Review action failed: {response.text}")
        
        # Try with our original attorney ID if we used a different one
        if action_attorney_id != attorney_id:
            print(f"\n   🔄 Retrying with original attorney ID: {attorney_id}")
            action_data["attorney_id"] = attorney_id
            response = requests.post(f"{api_url}/attorney/review/action", json=action_data)
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Review action successful with original attorney: {result}")
            else:
                print(f"   ❌ Review action still failed: {response.text}")

if __name__ == "__main__":
    debug_review_action()