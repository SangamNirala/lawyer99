import requests
import json

# Test HR compliance endpoint with correct parameters
base_url = "https://legal-research-api.preview.emergentagent.com"
api_url = f"{base_url}/api"

# Based on the error, it expects 'content' and 'content_type' fields
compliance_data_corrected = {
    "content": """
    EMPLOYMENT AGREEMENT
    
    This Employment Agreement is entered into between TechCorp Inc. and Jane Doe.
    
    **POSITION AND DUTIES**
    Employee will serve as Senior Software Engineer in the Engineering Department.
    
    **COMPENSATION**
    Base salary: $95,000 annually
    Benefits: Health insurance, dental, vision, 401k matching up to 4%
    
    **EMPLOYMENT TERMS**
    Employment Type: Full-time, at-will employment
    Start Date: February 1, 2025
    Probationary Period: 90 days
    
    **CONFIDENTIALITY**
    Employee agrees to maintain confidentiality of all proprietary information.
    
    **TERMINATION**
    Either party may terminate this agreement with 30 days written notice.
    """,
    "content_type": "employment_agreement",
    "jurisdiction": "US",
    "compliance_areas": ["employment_law", "benefits", "confidentiality"]
}

print("🔍 Testing HR Compliance with corrected parameters...")
print(f"URL: {api_url}/hr/compliance")
print(f"Request Data: {json.dumps(compliance_data_corrected, indent=2)}")

try:
    response = requests.post(
        f"{api_url}/hr/compliance",
        json=compliance_data_corrected,
        headers={'Content-Type': 'application/json'},
        timeout=30
    )
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        print("✅ HR Compliance endpoint working with correct parameters!")
        response_data = response.json()
        print(f"Response keys: {list(response_data.keys())}")
        
        # Show some key details
        if 'compliance_score' in response_data:
            print(f"Compliance Score: {response_data['compliance_score']}")
        if 'issues' in response_data:
            print(f"Issues Found: {len(response_data['issues'])}")
        if 'recommendations' in response_data:
            print(f"Recommendations: {len(response_data['recommendations'])}")
            
    else:
        print(f"❌ Still failing with status: {response.status_code}")
        try:
            error_data = response.json()
            print(f"Error: {error_data}")
        except:
            print(f"Error: {response.text}")
            
except Exception as e:
    print(f"❌ Request failed: {str(e)}")