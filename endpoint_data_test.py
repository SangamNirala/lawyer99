import requests
import json

def test_edited_pdf_endpoint_data():
    """Test that the edited PDF endpoint receives and processes data correctly"""
    base_url = "https://027a3f35-a0bc-40cd-ad79-483d46c6126e.preview.emergentagent.com"
    api_url = f"{base_url}/api"
    
    # Create a simple test contract data structure
    test_contract = {
        "id": "test-contract-123",
        "contract_type": "NDA",
        "jurisdiction": "US",
        "content": "**TEST CONTRACT**\n\nThis is a simple test contract for edited PDF generation.\n\n**EDITED CONTENT**: This text should appear in the PDF.",
        "compliance_score": 85.0,
        "created_at": "2025-01-30T05:00:00Z",
        "clauses": []
    }
    
    # Test the edited PDF endpoint
    edited_pdf_data = {"contract": test_contract}
    
    print("🔍 Testing edited PDF endpoint with simple test data...")
    print(f"   Contract ID: {test_contract['id']}")
    print(f"   Content preview: {test_contract['content'][:100]}...")
    
    try:
        response = requests.post(
            f"{api_url}/contracts/download-pdf-edited", 
            json=edited_pdf_data, 
            headers={'Content-Type': 'application/json'}, 
            timeout=30
        )
        
        print(f"   Response Status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Edited PDF endpoint responded successfully")
            
            # Check headers
            content_type = response.headers.get('content-type', '')
            content_disposition = response.headers.get('content-disposition', '')
            
            print(f"   Content-Type: {content_type}")
            print(f"   Content-Disposition: {content_disposition}")
            
            # Verify PDF format
            if response.content.startswith(b'%PDF'):
                print("   ✅ Valid PDF format")
            else:
                print("   ❌ Invalid PDF format")
                return False
            
            # Check PDF size
            pdf_size = len(response.content)
            print(f"   PDF Size: {pdf_size} bytes")
            
            if pdf_size > 500:  # Should be reasonably sized
                print("   ✅ PDF has reasonable size")
            else:
                print("   ❌ PDF seems too small")
                return False
            
            # Check filename
            if '_edited.pdf' in content_disposition:
                print("   ✅ Filename includes '_edited' suffix")
            else:
                print("   ❌ Filename missing '_edited' suffix")
            
            # Try to find our test content in the PDF
            try:
                pdf_content = response.content.decode('latin-1', errors='ignore')
                
                # Look for various indicators
                indicators_found = []
                test_indicators = [
                    'Edited',  # Status indicator
                    'TEST CONTRACT',  # Title
                    'EDITED CONTENT',  # Our test content
                    'simple test contract',  # Part of our content
                    test_contract['id']  # Contract ID
                ]
                
                for indicator in test_indicators:
                    if indicator in pdf_content:
                        indicators_found.append(indicator)
                
                print(f"   Content indicators found: {indicators_found}")
                
                if 'Edited' in indicators_found:
                    print("   ✅ 'Edited' status found in PDF")
                else:
                    print("   ❌ 'Edited' status NOT found in PDF")
                
                if len(indicators_found) >= 2:  # At least some content found
                    print("   ✅ Test content found in PDF")
                    return True
                else:
                    print("   ❌ Test content NOT found in PDF")
                    return False
                    
            except Exception as e:
                print(f"   ❌ Error analyzing PDF content: {str(e)}")
                return False
                
        else:
            print(f"❌ Edited PDF endpoint failed: {response.status_code}")
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
    success = test_edited_pdf_endpoint_data()
    if success:
        print("\n🎉 Edited PDF endpoint data test PASSED!")
    else:
        print("\n❌ Edited PDF endpoint data test FAILED!")