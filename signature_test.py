#!/usr/bin/env python3

import requests
import sys
import json
from datetime import datetime

class SignatureAPITester:
    def __init__(self, base_url="https://cad694cf-d3ea-4711-9a72-668a6a40e52f.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.contract_id = None

    def run_test(self, name, method, endpoint, expected_status, data=None, timeout=30):
        """Run a single API test"""
        url = f"{self.api_url}/{endpoint}" if not endpoint.startswith('http') else endpoint
        headers = {'Content-Type': 'application/json'}

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=timeout)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=timeout)

            print(f"   Status: {response.status_code}")
            
            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    response_data = response.json()
                    return True, response_data
                except:
                    return True, response.text
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data}")
                except:
                    print(f"   Error: {response.text}")
                return False, {}

        except requests.exceptions.Timeout:
            print(f"❌ Failed - Request timed out after {timeout} seconds")
            return False, {}
        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def test_contract_generation_with_signatures(self):
        """Test that generated contracts include signature sections"""
        signature_test_data = {
            "contract_type": "NDA",
            "jurisdiction": "US",
            "parties": {
                "party1_name": "Signature Test Corp",
                "party1_type": "corporation",
                "party2_name": "Digital Signature Tester",
                "party2_type": "individual"
            },
            "terms": {
                "purpose": "Testing digital signature functionality in contract generation",
                "duration": "2_years"
            },
            "special_clauses": ["Digital signature verification clause"]
        }
        
        success, response = self.run_test(
            "Generate Contract for Signature Testing",
            "POST",
            "generate-contract",
            200,
            signature_test_data,
            timeout=60
        )
        
        if success and 'contract' in response:
            contract = response['contract']
            content = contract.get('content', '')
            self.contract_id = contract.get('id')
            
            print(f"   Generated contract ID: {self.contract_id}")
            print(f"   Testing signature section requirements...")
            
            # Check for required signature elements
            signature_requirements = [
                "**SIGNATURES**",
                "IN WITNESS WHEREOF",
                "First Party Signature Placeholder",
                "Second Party Signature Placeholder"
            ]
            
            missing_elements = []
            for requirement in signature_requirements:
                if requirement not in content:
                    missing_elements.append(requirement)
                else:
                    print(f"   ✅ Found: {requirement}")
            
            if not missing_elements:
                print(f"   ✅ All signature section requirements met")
            else:
                print(f"   ❌ Missing signature elements: {missing_elements}")
            
            # Check for party names in signature sections
            party1_name = signature_test_data['parties']['party1_name']
            party2_name = signature_test_data['parties']['party2_name']
            
            if party1_name in content and party2_name in content:
                print(f"   ✅ Party names properly inserted in signature sections")
                print(f"     - {party1_name}: Found")
                print(f"     - {party2_name}: Found")
            else:
                print(f"   ❌ Party names not properly inserted in signature sections")
                if party1_name not in content:
                    print(f"     - {party1_name}: Missing")
                if party2_name not in content:
                    print(f"     - {party2_name}: Missing")
            
            # Show signature section preview
            signature_start = content.find("**SIGNATURES**")
            if signature_start != -1:
                signature_section = content[signature_start:signature_start+500]
                print(f"   Signature section preview:")
                print(f"   {signature_section[:300]}...")
        
        return success, response

    def test_signature_upload_valid_data(self):
        """Test signature upload endpoint with valid data"""
        if not self.contract_id:
            print("⚠️  Skipping signature upload test - no contract ID available")
            return True, {}
        
        # Create a simple base64 encoded test signature (1x1 pixel PNG)
        test_signature_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAI9jU77zgAAAABJRU5ErkJggg=="
        
        # Test first party signature upload
        first_party_data = {
            "contract_id": self.contract_id,
            "party_type": "first_party",
            "signature_image": test_signature_base64
        }
        
        success_first, response_first = self.run_test(
            "Upload First Party Signature",
            "POST",
            f"contracts/{self.contract_id}/upload-signature",
            200,
            first_party_data
        )
        
        if success_first:
            print(f"   ✅ First party signature uploaded successfully")
            print(f"   Response: {response_first}")
        
        # Test second party signature upload
        second_party_data = {
            "contract_id": self.contract_id,
            "party_type": "second_party", 
            "signature_image": test_signature_base64
        }
        
        success_second, response_second = self.run_test(
            "Upload Second Party Signature",
            "POST",
            f"contracts/{self.contract_id}/upload-signature",
            200,
            second_party_data
        )
        
        if success_second:
            print(f"   ✅ Second party signature uploaded successfully")
            print(f"   Response: {response_second}")
        
        return success_first and success_second, {
            "first_party": response_first,
            "second_party": response_second
        }

    def test_signature_upload_invalid_data(self):
        """Test signature upload endpoint with invalid data"""
        if not self.contract_id:
            print("⚠️  Skipping invalid signature upload test - no contract ID available")
            return True, {}
        
        # Test invalid party type
        invalid_party_data = {
            "contract_id": self.contract_id,
            "party_type": "invalid_party",
            "signature_image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAI9jU77zgAAAABJRU5ErkJggg=="
        }
        
        success_invalid_party, response_invalid_party = self.run_test(
            "Upload Signature with Invalid Party Type",
            "POST",
            f"contracts/{self.contract_id}/upload-signature",
            400,
            invalid_party_data
        )
        
        # Test missing signature data
        missing_signature_data = {
            "contract_id": self.contract_id,
            "party_type": "first_party",
            "signature_image": ""
        }
        
        success_missing_sig, response_missing_sig = self.run_test(
            "Upload Signature with Missing Image",
            "POST",
            f"contracts/{self.contract_id}/upload-signature",
            400,
            missing_signature_data
        )
        
        # Test invalid base64 data
        invalid_base64_data = {
            "contract_id": self.contract_id,
            "party_type": "first_party",
            "signature_image": "invalid-base64-data"
        }
        
        success_invalid_base64, response_invalid_base64 = self.run_test(
            "Upload Signature with Invalid Base64",
            "POST",
            f"contracts/{self.contract_id}/upload-signature",
            400,
            invalid_base64_data
        )
        
        return success_invalid_party and success_missing_sig and success_invalid_base64, {
            "invalid_party": response_invalid_party,
            "missing_signature": response_missing_sig,
            "invalid_base64": response_invalid_base64
        }

    def test_signature_retrieval(self):
        """Test signature retrieval endpoint"""
        if not self.contract_id:
            print("⚠️  Skipping signature retrieval test - no contract ID available")
            return True, {}
        
        success, response = self.run_test(
            "Get Contract Signatures",
            "GET",
            f"contracts/{self.contract_id}/signatures",
            200
        )
        
        if success:
            print(f"   Contract ID: {response.get('contract_id')}")
            print(f"   First Party Signature Present: {'Yes' if response.get('first_party_signature') else 'No'}")
            print(f"   Second Party Signature Present: {'Yes' if response.get('second_party_signature') else 'No'}")
            
            # Verify response structure
            expected_keys = ['contract_id', 'first_party_signature', 'second_party_signature']
            missing_keys = [key for key in expected_keys if key not in response]
            if not missing_keys:
                print(f"   ✅ Response contains all expected keys")
            else:
                print(f"   ❌ Missing keys in response: {missing_keys}")
        
        return success, response

    def test_pdf_generation_with_signatures(self):
        """Test PDF generation with uploaded signatures"""
        if not self.contract_id:
            print("⚠️  Skipping PDF signature test - no contract ID available")
            return True, {}
        
        # Test original PDF download with signatures
        pdf_url = f"{self.api_url}/contracts/{self.contract_id}/download-pdf"
        
        self.tests_run += 1
        print(f"\n🔍 Testing PDF Generation with Signatures...")
        print(f"   URL: {pdf_url}")
        
        try:
            response = requests.get(pdf_url, timeout=30)
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                self.tests_passed += 1
                print(f"✅ PDF with signatures generated successfully")
                
                # Verify PDF format and headers
                content_type = response.headers.get('content-type', '')
                if 'application/pdf' in content_type:
                    print("   ✅ Correct PDF content type")
                
                if response.content.startswith(b'%PDF'):
                    print("   ✅ Valid PDF format")
                
                content_length = len(response.content)
                print(f"   PDF Size: {content_length} bytes")
                
                if content_length > 2000:
                    print("   ✅ PDF has reasonable size (likely contains signature images)")
                
                # Try to verify signature content in PDF
                try:
                    pdf_content_str = response.content.decode('latin-1', errors='ignore')
                    
                    # Look for signature-related content
                    signature_indicators = ['SIGNATURES', 'FIRST PARTY', 'SECOND PARTY', 'IN WITNESS WHEREOF']
                    found_indicators = [indicator for indicator in signature_indicators if indicator in pdf_content_str]
                    
                    if found_indicators:
                        print(f"   ✅ Signature section found in PDF: {found_indicators}")
                    else:
                        print(f"   ⚠️  Could not verify signature section in PDF")
                    
                    # Look for evidence of image content (signatures)
                    if '/Image' in pdf_content_str or 'PNG' in pdf_content_str or 'JPEG' in pdf_content_str:
                        print(f"   ✅ Evidence of signature images found in PDF")
                    else:
                        print(f"   ⚠️  No clear evidence of signature images in PDF")
                        
                except Exception as e:
                    print(f"   ⚠️  Could not analyze PDF content: {str(e)}")
                
                return True, {"pdf_size": content_length, "contract_id": self.contract_id}
            else:
                print(f"❌ Failed - Expected 200, got {response.status_code}")
                return False, {}
                
        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def test_edited_pdf_with_signatures(self):
        """Test edited PDF generation with signature data included"""
        if not self.contract_id:
            print("⚠️  Skipping edited PDF signature test - no contract ID available")
            return True, {}
        
        # Get the contract data
        contract_success, contract_response = self.run_test(
            "Get Contract for Edited PDF Signature Test",
            "GET",
            f"contracts/{self.contract_id}",
            200
        )
        
        if not contract_success:
            print("❌ Failed to get contract for edited PDF signature test")
            return False, {}
        
        # Modify the contract content
        edited_contract = contract_response.copy()
        edited_contract['content'] = edited_contract['content'].replace(
            "Testing digital signature functionality in contract generation",
            "EDITED: Testing digital signature functionality in contract generation with modified content"
        )
        
        # Test edited PDF generation
        edited_pdf_data = {"contract": edited_contract}
        edited_pdf_url = f"{self.api_url}/contracts/download-pdf-edited"
        
        self.tests_run += 1
        print(f"\n🔍 Testing Edited PDF Generation with Signatures...")
        print(f"   URL: {edited_pdf_url}")
        
        try:
            response = requests.post(
                edited_pdf_url,
                json=edited_pdf_data,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                self.tests_passed += 1
                print(f"✅ Edited PDF with signatures generated successfully")
                
                # Verify PDF format
                if response.content.startswith(b'%PDF'):
                    print("   ✅ Valid PDF format")
                
                content_length = len(response.content)
                print(f"   PDF Size: {content_length} bytes")
                
                # Check filename includes 'edited'
                content_disposition = response.headers.get('content-disposition', '')
                if '_edited.pdf' in content_disposition:
                    print("   ✅ Filename includes 'edited' indicator")
                
                # Try to verify both edited content and signatures in PDF
                try:
                    pdf_content_str = response.content.decode('latin-1', errors='ignore')
                    
                    # Check for edited content
                    if 'EDITED:' in pdf_content_str:
                        print("   ✅ Edited content found in PDF")
                    
                    # Check for signature sections
                    signature_indicators = ['SIGNATURES', 'FIRST PARTY', 'SECOND PARTY']
                    found_indicators = [indicator for indicator in signature_indicators if indicator in pdf_content_str]
                    
                    if found_indicators:
                        print(f"   ✅ Signature sections preserved in edited PDF: {found_indicators}")
                    
                    # Check for 'Edited' status indicator
                    if 'Status:</b> Edited' in pdf_content_str or 'Status: Edited' in pdf_content_str:
                        print("   ✅ 'Edited' status indicator found in PDF metadata")
                    
                except Exception as e:
                    print(f"   ⚠️  Could not analyze edited PDF content: {str(e)}")
                
                return True, {"pdf_size": content_length}
            else:
                print(f"❌ Failed - Expected 200, got {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data}")
                except:
                    print(f"   Error: {response.text}")
                return False, {}
                
        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def run_signature_tests(self):
        """Run all signature-related tests"""
        print("🖋️  DIGITAL SIGNATURE FUNCTIONALITY TESTS")
        print("=" * 80)
        
        # Test 1: Contract Generation with Signatures
        self.test_contract_generation_with_signatures()
        
        # Test 2: Signature Upload (Valid Data)
        self.test_signature_upload_valid_data()
        
        # Test 3: Signature Upload (Invalid Data)
        self.test_signature_upload_invalid_data()
        
        # Test 4: Signature Retrieval
        self.test_signature_retrieval()
        
        # Test 5: PDF Generation with Signatures
        self.test_pdf_generation_with_signatures()
        
        # Test 6: Edited PDF with Signatures
        self.test_edited_pdf_with_signatures()
        
        # Print summary
        print("\n" + "=" * 80)
        print("📊 SIGNATURE TEST SUMMARY")
        print("=" * 80)
        print(f"Total Tests Run: {self.tests_run}")
        print(f"Tests Passed: {self.tests_passed}")
        print(f"Tests Failed: {self.tests_run - self.tests_passed}")
        print(f"Success Rate: {(self.tests_passed / self.tests_run * 100):.1f}%")
        
        if self.tests_passed == self.tests_run:
            print("🎉 All signature tests passed!")
        else:
            print(f"⚠️  {self.tests_run - self.tests_passed} signature test(s) failed")
        
        return self.tests_passed == self.tests_run

def main():
    print("🚀 Starting Digital Signature API Tests")
    print("=" * 60)
    
    tester = SignatureAPITester()
    success = tester.run_signature_tests()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())