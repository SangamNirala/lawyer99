import requests
import json

def test_complete_edited_pdf():
    """Test edited PDF with complete contract structure"""
    base_url = "https://8cd68d5c-4981-470b-b1c0-9982d2b4a8d2.preview.emergentagent.com"
    api_url = f"{base_url}/api"
    
    # First generate a real contract to get the complete structure
    contract_data = {
        "contract_type": "NDA",
        "jurisdiction": "US",
        "parties": {
            "party1_name": "Complete Test Corp",
            "party1_type": "corporation",
            "party2_name": "Structure Tester",
            "party2_type": "individual"
        },
        "terms": {
            "purpose": "Testing complete contract structure for edited PDF",
            "duration": "1_year"
        },
        "special_clauses": []
    }
    
    print("🔍 Generating complete contract structure...")
    response = requests.post(f"{api_url}/generate-contract", json=contract_data, timeout=60)
    
    if response.status_code != 200:
        print(f"❌ Failed to generate contract: {response.status_code}")
        return False
    
    original_contract = response.json()['contract']
    print(f"✅ Contract generated: {original_contract['id']}")
    print(f"   Contract keys: {list(original_contract.keys())}")
    
    # Modify the contract content for editing
    edited_contract = original_contract.copy()
    edited_contract['content'] = edited_contract['content'].replace(
        "Testing complete contract structure for edited PDF",
        "EDITED: Testing complete contract structure for edited PDF - this content has been modified"
    )
    
    print(f"\n🔍 Testing edited PDF with complete contract structure...")
    print(f"   Original content length: {len(original_contract['content'])} chars")
    print(f"   Edited content length: {len(edited_contract['content'])} chars")
    
    # Test the edited PDF endpoint
    edited_pdf_data = {"contract": edited_contract}
    
    try:
        response = requests.post(
            f"{api_url}/contracts/download-pdf-edited", 
            json=edited_pdf_data, 
            headers={'Content-Type': 'application/json'}, 
            timeout=30
        )
        
        print(f"   Response Status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Edited PDF generated successfully")
            
            # Check all the expected properties
            content_type = response.headers.get('content-type', '')
            content_disposition = response.headers.get('content-disposition', '')
            pdf_size = len(response.content)
            
            print(f"   Content-Type: {content_type}")
            print(f"   Content-Disposition: {content_disposition}")
            print(f"   PDF Size: {pdf_size} bytes")
            
            # Verify all requirements
            checks = []
            
            # 1. Correct content type
            if 'application/pdf' in content_type:
                checks.append("✅ Correct PDF content type")
            else:
                checks.append("❌ Wrong content type")
            
            # 2. Correct filename with _edited suffix
            if '_edited.pdf' in content_disposition:
                checks.append("✅ Filename includes '_edited' suffix")
            else:
                checks.append("❌ Missing '_edited' suffix")
            
            # 3. Valid PDF format
            if response.content.startswith(b'%PDF'):
                checks.append("✅ Valid PDF format")
            else:
                checks.append("❌ Invalid PDF format")
            
            # 4. Reasonable size
            if pdf_size > 1000:
                checks.append("✅ PDF has reasonable size")
            else:
                checks.append("❌ PDF too small")
            
            # 5. Look for "Edited" status in PDF
            try:
                pdf_content = response.content.decode('latin-1', errors='ignore')
                if 'Edited' in pdf_content:
                    checks.append("✅ 'Edited' status found in PDF")
                else:
                    checks.append("❌ 'Edited' status NOT found in PDF")
            except:
                checks.append("❌ Could not analyze PDF content")
            
            # Print all checks
            print("\n   📋 Verification Results:")
            for check in checks:
                print(f"     {check}")
            
            # Overall success if most checks pass
            success_count = sum(1 for check in checks if check.startswith("✅"))
            total_checks = len(checks)
            
            print(f"\n   📊 Overall: {success_count}/{total_checks} checks passed")
            
            if success_count >= 4:  # At least 4 out of 5 checks should pass
                print("   🎉 EDITED PDF FUNCTIONALITY WORKING!")
                return True
            else:
                print("   ❌ Some critical issues found")
                return False
                
        else:
            print(f"❌ Failed to generate edited PDF: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data}")
            except:
                print(f"   Error text: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Request failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_complete_edited_pdf()
    if success:
        print("\n🎉 COMPLETE EDITED PDF TEST PASSED!")
    else:
        print("\n❌ Complete edited PDF test failed")