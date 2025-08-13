import requests
import json

def compare_original_vs_edited_pdf():
    """Compare original PDF vs edited PDF to verify differences"""
    base_url = "https://legalcore.preview.emergentagent.com"
    api_url = f"{base_url}/api"
    
    # Generate a contract
    contract_data = {
        "contract_type": "NDA",
        "jurisdiction": "US",
        "parties": {
            "party1_name": "Compare Test Corp",
            "party1_type": "corporation",
            "party2_name": "PDF Comparison Tester",
            "party2_type": "individual"
        },
        "terms": {
            "purpose": "Testing PDF comparison functionality",
            "duration": "1_year"
        },
        "special_clauses": []
    }
    
    print("🔍 Generating contract for comparison test...")
    response = requests.post(f"{api_url}/generate-contract", json=contract_data, timeout=60)
    
    if response.status_code != 200:
        print(f"❌ Failed to generate contract: {response.status_code}")
        return False
    
    contract = response.json()['contract']
    contract_id = contract['id']
    print(f"✅ Contract generated: {contract_id}")
    
    # Download original PDF
    print("🔍 Downloading original PDF...")
    original_pdf_response = requests.get(f"{api_url}/contracts/{contract_id}/download-pdf", timeout=30)
    
    if original_pdf_response.status_code != 200:
        print(f"❌ Failed to download original PDF: {original_pdf_response.status_code}")
        return False
    
    original_pdf_size = len(original_pdf_response.content)
    print(f"✅ Original PDF downloaded: {original_pdf_size} bytes")
    
    # Create edited version
    edited_contract = contract.copy()
    edited_contract['content'] = edited_contract['content'].replace(
        "Testing PDF comparison functionality",
        "EDITED: Testing PDF comparison functionality - this content has been modified"
    )
    
    # Generate edited PDF
    print("🔍 Generating edited PDF...")
    edited_pdf_data = {"contract": edited_contract}
    edited_pdf_response = requests.post(
        f"{api_url}/contracts/download-pdf-edited", 
        json=edited_pdf_data, 
        headers={'Content-Type': 'application/json'}, 
        timeout=30
    )
    
    if edited_pdf_response.status_code != 200:
        print(f"❌ Failed to generate edited PDF: {edited_pdf_response.status_code}")
        print(f"   Error: {edited_pdf_response.text}")
        return False
    
    edited_pdf_size = len(edited_pdf_response.content)
    print(f"✅ Edited PDF generated: {edited_pdf_size} bytes")
    
    # Compare headers
    print("\n📋 Comparing PDF headers:")
    print(f"   Original Content-Disposition: {original_pdf_response.headers.get('content-disposition', 'N/A')}")
    print(f"   Edited Content-Disposition: {edited_pdf_response.headers.get('content-disposition', 'N/A')}")
    
    # Check if edited filename is different
    original_filename = original_pdf_response.headers.get('content-disposition', '')
    edited_filename = edited_pdf_response.headers.get('content-disposition', '')
    
    if '_edited.pdf' in edited_filename and '_edited.pdf' not in original_filename:
        print("   ✅ Edited PDF has correct filename with '_edited' suffix")
    else:
        print("   ❌ Edited PDF filename issue")
    
    # Compare content sizes
    size_diff = abs(edited_pdf_size - original_pdf_size)
    print(f"\n📊 Size comparison:")
    print(f"   Original: {original_pdf_size} bytes")
    print(f"   Edited: {edited_pdf_size} bytes")
    print(f"   Difference: {size_diff} bytes")
    
    if size_diff > 10:  # Should be some difference due to "Edited" status
        print("   ✅ PDFs have different sizes (expected)")
    else:
        print("   ⚠️  PDFs have very similar sizes")
    
    # Check PDF format validity
    original_valid = original_pdf_response.content.startswith(b'%PDF')
    edited_valid = edited_pdf_response.content.startswith(b'%PDF')
    
    print(f"\n📄 PDF format validation:")
    print(f"   Original PDF valid: {'✅' if original_valid else '❌'}")
    print(f"   Edited PDF valid: {'✅' if edited_valid else '❌'}")
    
    # Try to find differences in raw content
    try:
        original_content = original_pdf_response.content.decode('latin-1', errors='ignore')
        edited_content = edited_pdf_response.content.decode('latin-1', errors='ignore')
        
        # Look for "Edited" in the edited PDF but not in original
        edited_in_original = 'Edited' in original_content
        edited_in_edited = 'Edited' in edited_content
        
        print(f"\n🔍 Content analysis:")
        print(f"   'Edited' found in original PDF: {'❌ Yes (unexpected)' if edited_in_original else '✅ No (expected)'}")
        print(f"   'Edited' found in edited PDF: {'✅ Yes (expected)' if edited_in_edited else '❌ No (missing)'}")
        
        # Look for the modified content
        modified_text_in_edited = 'EDITED: Testing PDF comparison functionality' in edited_content
        print(f"   Modified content in edited PDF: {'✅ Yes' if modified_text_in_edited else '❌ No'}")
        
        success = (not edited_in_original and edited_in_edited and 
                  original_valid and edited_valid and 
                  '_edited.pdf' in edited_filename)
        
        return success
        
    except Exception as e:
        print(f"❌ Error analyzing content: {str(e)}")
        return False

if __name__ == "__main__":
    success = compare_original_vs_edited_pdf()
    if success:
        print("\n🎉 PDF comparison test PASSED!")
    else:
        print("\n❌ PDF comparison test had issues - check details above")