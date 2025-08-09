#!/usr/bin/env python3
"""
Focused test to reproduce the signature bug mentioned in the review request.
This test specifically tests the signature functionality with the provided test images.
"""

import requests
import base64
import sys
import json
from datetime import datetime

class SignatureBugTester:
    def __init__(self, base_url="https://cad694cf-d3ea-4711-9a72-668a6a40e52f.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.contract_id = None

    def download_and_encode_image(self, url):
        """Download image from URL and convert to base64"""
        try:
            print(f"   📥 Downloading image from: {url}")
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                # Convert to base64
                image_base64 = base64.b64encode(response.content).decode('utf-8')
                print(f"   ✅ Image downloaded and encoded ({len(image_base64)} chars)")
                return image_base64
            else:
                print(f"   ❌ Failed to download image: {response.status_code}")
                return None
        except Exception as e:
            print(f"   ❌ Error downloading image: {str(e)}")
            return None

    def create_test_contract(self):
        """Create a new contract for signature testing"""
        print("\n🔍 Step 1: Creating a new contract...")
        
        contract_data = {
            "contract_type": "NDA",
            "jurisdiction": "US",
            "parties": {
                "party1_name": "Signature Test Corp",
                "party1_type": "corporation",
                "party2_name": "Digital Signature Tester",
                "party2_type": "individual"
            },
            "terms": {
                "purpose": "Testing signature functionality to reproduce the PDF download bug",
                "duration": "2_years"
            },
            "special_clauses": ["Signature verification clause"]
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/generate-contract",
                json=contract_data,
                headers={'Content-Type': 'application/json'},
                timeout=60
            )
            
            if response.status_code == 200:
                contract_response = response.json()
                self.contract_id = contract_response['contract']['id']
                print(f"   ✅ Contract created successfully")
                print(f"   📄 Contract ID: {self.contract_id}")
                return True
            else:
                print(f"   ❌ Failed to create contract: {response.status_code}")
                print(f"   Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"   ❌ Error creating contract: {str(e)}")
            return False

    def upload_signatures(self):
        """Upload signatures using the provided test images"""
        print("\n🔍 Step 2: Uploading signatures with test images...")
        
        # Test image URLs from the review request
        first_party_url = "https://customer-assets.emergentagent.com/job_sigrender/artifacts/c3ul5u5b_sign1.jpeg"
        second_party_url = "https://customer-assets.emergentagent.com/job_sigrender/artifacts/ko66f06n_sign2.png"
        
        # Download and encode first party signature
        print("\n   📝 Uploading First Party Signature...")
        first_party_base64 = self.download_and_encode_image(first_party_url)
        if not first_party_base64:
            return False
        
        # Upload first party signature
        first_party_data = {
            "contract_id": self.contract_id,
            "party_type": "first_party",
            "signature_image": first_party_base64
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/contracts/{self.contract_id}/upload-signature",
                json=first_party_data,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if response.status_code == 200:
                print(f"   ✅ First party signature uploaded successfully")
            else:
                print(f"   ❌ Failed to upload first party signature: {response.status_code}")
                print(f"   Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"   ❌ Error uploading first party signature: {str(e)}")
            return False
        
        # Download and encode second party signature
        print("\n   📝 Uploading Second Party Signature...")
        second_party_base64 = self.download_and_encode_image(second_party_url)
        if not second_party_base64:
            return False
        
        # Upload second party signature
        second_party_data = {
            "contract_id": self.contract_id,
            "party_type": "second_party",
            "signature_image": second_party_base64
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/contracts/{self.contract_id}/upload-signature",
                json=second_party_data,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if response.status_code == 200:
                print(f"   ✅ Second party signature uploaded successfully")
                return True
            else:
                print(f"   ❌ Failed to upload second party signature: {response.status_code}")
                print(f"   Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"   ❌ Error uploading second party signature: {str(e)}")
            return False

    def test_signature_retrieval(self):
        """Test the signature retrieval endpoint"""
        print("\n🔍 Step 3: Testing signature retrieval...")
        
        try:
            response = requests.get(
                f"{self.api_url}/contracts/{self.contract_id}/signatures",
                timeout=30
            )
            
            if response.status_code == 200:
                signatures = response.json()
                print(f"   ✅ Signatures retrieved successfully")
                print(f"   📄 Contract ID: {signatures.get('contract_id')}")
                
                first_party_sig = signatures.get('first_party_signature')
                second_party_sig = signatures.get('second_party_signature')
                
                if first_party_sig:
                    print(f"   ✅ First party signature present ({len(first_party_sig)} chars)")
                else:
                    print(f"   ❌ First party signature missing")
                
                if second_party_sig:
                    print(f"   ✅ Second party signature present ({len(second_party_sig)} chars)")
                else:
                    print(f"   ❌ Second party signature missing")
                
                return first_party_sig and second_party_sig
            else:
                print(f"   ❌ Failed to retrieve signatures: {response.status_code}")
                print(f"   Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"   ❌ Error retrieving signatures: {str(e)}")
            return False

    def test_pdf_download_original(self):
        """Test original contract PDF download"""
        print("\n🔍 Step 4: Testing original contract PDF download...")
        
        try:
            response = requests.get(
                f"{self.api_url}/contracts/{self.contract_id}/download-pdf",
                timeout=30
            )
            
            if response.status_code == 200:
                print(f"   ✅ PDF download successful")
                print(f"   📄 Content-Type: {response.headers.get('content-type')}")
                print(f"   📄 Content-Disposition: {response.headers.get('content-disposition')}")
                print(f"   📄 PDF Size: {len(response.content)} bytes")
                
                # Check if it's a valid PDF
                if response.content.startswith(b'%PDF'):
                    print(f"   ✅ Valid PDF format")
                else:
                    print(f"   ❌ Invalid PDF format")
                    return False
                
                # Analyze PDF content for signature issues
                try:
                    pdf_content_str = response.content.decode('latin-1', errors='ignore')
                    
                    # Check for signature error messages
                    if '[Signature Image Error]' in pdf_content_str:
                        print(f"   🚨 BUG REPRODUCED: '[Signature Image Error]' found in PDF!")
                        print(f"   🚨 This confirms the reported bug where signatures show as error in PDF")
                        return "BUG_REPRODUCED"
                    else:
                        print(f"   ✅ No '[Signature Image Error]' found in PDF")
                    
                    # Check for signature placeholders (should not be present)
                    placeholder_issues = []
                    if '[First Party Signature Placeholder]' in pdf_content_str:
                        placeholder_issues.append("First Party Placeholder found")
                    if '[First Party Signature Uploaded]' in pdf_content_str:
                        placeholder_issues.append("First Party Uploaded placeholder found")
                    if '[Second Party Signature Placeholder]' in pdf_content_str:
                        placeholder_issues.append("Second Party Placeholder found")
                    if '[Second Party Signature Uploaded]' in pdf_content_str:
                        placeholder_issues.append("Second Party Uploaded placeholder found")
                    
                    if placeholder_issues:
                        print(f"   🚨 SIGNATURE PLACEHOLDER ISSUES:")
                        for issue in placeholder_issues:
                            print(f"      - {issue}")
                        return "PLACEHOLDER_ISSUES"
                    else:
                        print(f"   ✅ No signature placeholders found in PDF")
                    
                    # Check for evidence of signature images
                    image_indicators = ['Image', '/Image', 'PNG', 'JPEG', 'ImageReader']
                    found_images = [ind for ind in image_indicators if ind in pdf_content_str]
                    
                    if found_images:
                        print(f"   ✅ Signature images embedded in PDF: {found_images}")
                    else:
                        print(f"   ⚠️  No clear evidence of signature images in PDF")
                    
                    # Check for signature sections
                    signature_sections = ['SIGNATURES', 'FIRST PARTY', 'SECOND PARTY', 'IN WITNESS WHEREOF']
                    found_sections = [sec for sec in signature_sections if sec in pdf_content_str]
                    
                    if found_sections:
                        print(f"   ✅ Signature sections found: {found_sections}")
                    else:
                        print(f"   ❌ No signature sections found in PDF")
                    
                    return True
                    
                except Exception as e:
                    print(f"   ⚠️  Could not analyze PDF content: {str(e)}")
                    return True  # PDF download worked, just couldn't analyze
                
            else:
                print(f"   ❌ PDF download failed: {response.status_code}")
                print(f"   Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"   ❌ Error downloading PDF: {str(e)}")
            return False

    def test_pdf_download_edited(self):
        """Test edited contract PDF download"""
        print("\n🔍 Step 5: Testing edited contract PDF download...")
        
        # First get the contract data
        try:
            response = requests.get(
                f"{self.api_url}/contracts/{self.contract_id}",
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"   ❌ Failed to get contract data: {response.status_code}")
                return False
            
            contract_data = response.json()
            
            # Modify the contract content
            contract_data['content'] = contract_data['content'].replace(
                "Testing signature functionality to reproduce the PDF download bug",
                "EDITED: Testing signature functionality to reproduce the PDF download bug with modified content"
            )
            
            # Test edited PDF download
            edited_pdf_data = {"contract": contract_data}
            
            response = requests.post(
                f"{self.api_url}/contracts/download-pdf-edited",
                json=edited_pdf_data,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if response.status_code == 200:
                print(f"   ✅ Edited PDF download successful")
                print(f"   📄 Content-Type: {response.headers.get('content-type')}")
                print(f"   📄 Content-Disposition: {response.headers.get('content-disposition')}")
                print(f"   📄 PDF Size: {len(response.content)} bytes")
                
                # Check if it's a valid PDF
                if response.content.startswith(b'%PDF'):
                    print(f"   ✅ Valid PDF format")
                else:
                    print(f"   ❌ Invalid PDF format")
                    return False
                
                # Analyze edited PDF content for signature issues
                try:
                    pdf_content_str = response.content.decode('latin-1', errors='ignore')
                    
                    # Check for signature error messages
                    if '[Signature Image Error]' in pdf_content_str:
                        print(f"   🚨 BUG REPRODUCED: '[Signature Image Error]' found in edited PDF!")
                        return "BUG_REPRODUCED"
                    else:
                        print(f"   ✅ No '[Signature Image Error]' found in edited PDF")
                    
                    # Check for edited content
                    if 'EDITED:' in pdf_content_str:
                        print(f"   ✅ Edited content confirmed in PDF")
                    else:
                        print(f"   ⚠️  Could not verify edited content in PDF")
                    
                    # Check for signature images
                    image_indicators = ['Image', '/Image', 'PNG', 'JPEG', 'ImageReader']
                    found_images = [ind for ind in image_indicators if ind in pdf_content_str]
                    
                    if found_images:
                        print(f"   ✅ Signature images embedded in edited PDF: {found_images}")
                    else:
                        print(f"   ⚠️  No clear evidence of signature images in edited PDF")
                    
                    return True
                    
                except Exception as e:
                    print(f"   ⚠️  Could not analyze edited PDF content: {str(e)}")
                    return True
                
            else:
                print(f"   ❌ Edited PDF download failed: {response.status_code}")
                print(f"   Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"   ❌ Error with edited PDF download: {str(e)}")
            return False

    def check_backend_logs(self):
        """Check backend logs for signature processing errors"""
        print("\n🔍 Step 6: Checking backend logs for errors...")
        
        try:
            # This would typically check supervisor logs, but we'll simulate
            print("   📋 Backend logs would be checked here for:")
            print("      - Signature processing errors")
            print("      - Base64 decoding issues")
            print("      - PIL image processing errors")
            print("      - Reportlab integration problems")
            print("   ℹ️  In a real environment, check: tail -n 100 /var/log/supervisor/backend.*.log")
            return True
        except Exception as e:
            print(f"   ⚠️  Could not check backend logs: {str(e)}")
            return True

    def run_signature_bug_test(self):
        """Run the complete signature bug reproduction test"""
        print("🚀 Starting Signature Bug Reproduction Test")
        print("=" * 60)
        print("📋 Test Objective: Reproduce the bug where signatures appear correctly")
        print("   in frontend preview but show '[Signature Image Error]' in downloaded PDFs")
        print("=" * 60)
        
        # Step 1: Create contract
        if not self.create_test_contract():
            print("\n❌ Test failed at contract creation step")
            return False
        
        # Step 2: Upload signatures
        if not self.upload_signatures():
            print("\n❌ Test failed at signature upload step")
            return False
        
        # Step 3: Test signature retrieval
        if not self.test_signature_retrieval():
            print("\n❌ Test failed at signature retrieval step")
            return False
        
        # Step 4: Test original PDF download
        original_pdf_result = self.test_pdf_download_original()
        if original_pdf_result == "BUG_REPRODUCED":
            print("\n🚨 BUG SUCCESSFULLY REPRODUCED!")
            print("   The reported issue has been confirmed in original PDF download")
        elif original_pdf_result == "PLACEHOLDER_ISSUES":
            print("\n🚨 SIGNATURE PLACEHOLDER ISSUES FOUND!")
            print("   Signatures are not being processed correctly in PDF")
        elif not original_pdf_result:
            print("\n❌ Test failed at original PDF download step")
            return False
        
        # Step 5: Test edited PDF download
        edited_pdf_result = self.test_pdf_download_edited()
        if edited_pdf_result == "BUG_REPRODUCED":
            print("\n🚨 BUG ALSO REPRODUCED IN EDITED PDF!")
            print("   The issue affects both original and edited PDF downloads")
        elif not edited_pdf_result:
            print("\n❌ Test failed at edited PDF download step")
            return False
        
        # Step 6: Check backend logs
        self.check_backend_logs()
        
        # Final assessment
        print("\n" + "=" * 60)
        print("📊 SIGNATURE BUG TEST RESULTS")
        print("=" * 60)
        
        if original_pdf_result == "BUG_REPRODUCED" or edited_pdf_result == "BUG_REPRODUCED":
            print("🚨 BUG STATUS: REPRODUCED")
            print("   '[Signature Image Error]' found in downloaded PDFs")
            print("   This confirms the user's reported issue")
        elif original_pdf_result == "PLACEHOLDER_ISSUES":
            print("🚨 BUG STATUS: SIGNATURE PROCESSING ISSUES")
            print("   Signature placeholders not being replaced properly")
        else:
            print("✅ BUG STATUS: NOT REPRODUCED")
            print("   Signatures appear to be working correctly in PDFs")
            print("   The reported issue may have been fixed")
        
        print(f"\n📄 Test Contract ID: {self.contract_id}")
        print("   You can manually verify the results by:")
        print(f"   1. Visiting the frontend and viewing contract {self.contract_id}")
        print("   2. Downloading the PDF and checking for signature images")
        print("   3. Comparing frontend preview with downloaded PDF")
        
        return True

def main():
    tester = SignatureBugTester()
    success = tester.run_signature_bug_test()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())