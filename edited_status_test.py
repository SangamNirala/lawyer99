import requests
import json

def test_edited_status_in_pdf():
    """Test that the 'Edited' status appears in the PDF metadata"""
    base_url = "https://legal-research-api.preview.emergentagent.com"
    api_url = f"{base_url}/api"
    
    # Generate a contract first
    contract_data = {
        "contract_type": "NDA",
        "jurisdiction": "US",
        "parties": {
            "party1_name": "Status Test Corp",
            "party1_type": "corporation",
            "party2_name": "Edited Status Verifier",
            "party2_type": "individual"
        },
        "terms": {
            "purpose": "Testing edited status in PDF metadata",
            "duration": "1_year"
        },
        "special_clauses": []
    }
    
    print("🔍 Generating contract for edited status test...")
    response = requests.post(f"{api_url}/generate-contract", json=contract_data, timeout=60)
    
    if response.status_code != 200:
        print(f"❌ Failed to generate contract: {response.status_code}")
        return False
    
    contract = response.json()['contract']
    print(f"✅ Contract generated: {contract['id']}")
    
    # Create edited version
    edited_contract = contract.copy()
    edited_contract['content'] = edited_contract['content'].replace(
        "Testing edited status in PDF metadata",
        "EDITED: Testing edited status in PDF metadata - this content has been modified"
    )
    
    # Generate edited PDF
    edited_pdf_data = {"contract": edited_contract}
    
    print("🔍 Generating edited PDF...")
    pdf_response = requests.post(
        f"{api_url}/contracts/download-pdf-edited", 
        json=edited_pdf_data, 
        headers={'Content-Type': 'application/json'}, 
        timeout=30
    )
    
    if pdf_response.status_code != 200:
        print(f"❌ Failed to generate edited PDF: {pdf_response.status_code}")
        return False
    
    print(f"✅ Edited PDF generated successfully")
    print(f"   PDF Size: {len(pdf_response.content)} bytes")
    print(f"   Content-Type: {pdf_response.headers.get('content-type')}")
    print(f"   Content-Disposition: {pdf_response.headers.get('content-disposition')}")
    
    # Analyze PDF content for "Edited" status
    try:
        pdf_content = pdf_response.content.decode('latin-1', errors='ignore')
        
        # Look for various forms of "Edited" status
        edited_indicators = [
            'Status:</b> Edited',
            'Status: Edited',
            '<b>Status:</b> Edited',
            'Edited',
            'EDITED'
        ]
        
        found_indicators = []
        for indicator in edited_indicators:
            if indicator in pdf_content:
                found_indicators.append(indicator)
        
        if found_indicators:
            print(f"✅ Found 'Edited' status indicators in PDF: {found_indicators}")
            return True
        else:
            print("❌ Could not find 'Edited' status indicator in PDF")
            # Show a sample of the PDF content for debugging
            print("   PDF content sample (first 1000 chars):")
            print(f"   {pdf_content[:1000]}")
            return False
            
    except Exception as e:
        print(f"❌ Error analyzing PDF content: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_edited_status_in_pdf()
    if success:
        print("\n🎉 Edited status test PASSED!")
    else:
        print("\n❌ Edited status test FAILED!")