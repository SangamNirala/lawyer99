#!/usr/bin/env python3
"""
Focused signature functionality testing for LegalMate API
Testing the critical signature PDF fix and real signature images
"""

import requests
import sys
import json
import base64
from datetime import datetime

class SignatureFocusedTester:
    def __init__(self, base_url="https://legal-engine-check.preview.emergentagent.com"):
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

    def generate_test_contract(self):
        """Generate a contract for signature testing"""
        contract_data = {
            "contract_type": "NDA",
            "jurisdiction": "US",
            "parties": {
                "party1_name": "Signature Test Corp",
                "party1_type": "corporation",
                "party2_name": "Digital Signature Validator",
                "party2_type": "individual"
            },
            "terms": {
                "purpose": "Testing comprehensive signature functionality with real signature images",
                "duration": "2_years"
            },
            "special_clauses": ["Signature verification clause"]
        }
        
        success, response = self.run_test(
            "Generate Contract for Signature Testing",
            "POST",
            "generate-contract",
            200,
            contract_data,
            timeout=60
        )
        
        if success and 'contract' in response:
            contract = response['contract']
            self.contract_id = contract.get('id')
            print(f"   Contract ID: {self.contract_id}")
            return True, contract
        
        return False, {}

    def test_signature_upload_storage(self):
        """Test signature upload and storage functionality"""
        if not self.contract_id:
            print("⚠️  No contract ID available for signature upload test")
            return False, {}
        
        # Load real signature images
        try:
            with open('/app/sign1.jpeg', 'rb') as f:
                sign1_data = f.read()
                sign1_base64 = base64.b64encode(sign1_data).decode('utf-8')
            
            with open('/app/sign2.png', 'rb') as f:
                sign2_data = f.read()
                sign2_base64 = base64.b64encode(sign2_data).decode('utf-8')
            
            print(f"   ✅ Loaded sign1.jpeg: {len(sign1_data)} bytes")
            print(f"   ✅ Loaded sign2.png: {len(sign2_data)} bytes")
            
        except Exception as e:
            print(f"   ❌ Failed to load signature images: {str(e)}")
            return False, {}
        
        # Upload first party signature (sign1.jpeg)
        fp_sig_data = {
            "contract_id": self.contract_id,
            "party_type": "first_party",
            "signature_image": sign1_base64
        }
        
        success_fp, response_fp = self.run_test(
            "Upload First Party Signature (sign1.jpeg)",
            "POST",
            f"contracts/{self.contract_id}/upload-signature",
            200,
            fp_sig_data
        )
        
        # Upload second party signature (sign2.png)
        sp_sig_data = {
            "contract_id": self.contract_id,
            "party_type": "second_party",
            "signature_image": sign2_base64
        }
        
        success_sp, response_sp = self.run_test(
            "Upload Second Party Signature (sign2.png)",
            "POST",
            f"contracts/{self.contract_id}/upload-signature",
            200,
            sp_sig_data
        )
        
        return success_fp and success_sp, {
            "first_party": response_fp,
            "second_party": response_sp
        }

    def test_signature_retrieval(self):
        """Test signature retrieval functionality"""
        if not self.contract_id:
            print("⚠️  No contract ID available for signature retrieval test")
            return False, {}
        
        success, response = self.run_test(
            "Retrieve Contract Signatures",
            "GET",
            f"contracts/{self.contract_id}/signatures",
            200
        )
        
        if success:
            print(f"   Contract ID: {response.get('contract_id')}")
            fp_sig = response.get('first_party_signature')
            sp_sig = response.get('second_party_signature')
            
            if fp_sig:
                print(f"   ✅ First party signature retrieved: {len(fp_sig)} characters")
            else:
                print(f"   ❌ First party signature missing")
            
            if sp_sig:
                print(f"   ✅ Second party signature retrieved: {len(sp_sig)} characters")
            else:
                print(f"   ❌ Second party signature missing")
        
        return success, response

    def test_pdf_generation_with_signatures(self):
        """Test PDF generation with embedded signatures - CRITICAL TEST"""
        if not self.contract_id:
            print("⚠️  No contract ID available for PDF generation test")
            return False, {}
        
        pdf_url = f"{self.api_url}/contracts/{self.contract_id}/download-pdf"
        
        self.tests_run += 1
        print(f"\n🔍 CRITICAL TEST: PDF Generation with Real Signature Images...")
        print(f"   URL: {pdf_url}")
        print("   🎯 MAIN OBJECTIVE: Verify NO '[Signature Image Error]' messages appear")
        
        try:
            pdf_response = requests.get(pdf_url, timeout=30)
            print(f"   Status: {pdf_response.status_code}")
            
            if pdf_response.status_code == 200:
                self.tests_passed += 1
                print(f"✅ PDF download successful")
                
                # Verify PDF format
                if not pdf_response.content.startswith(b'%PDF'):
                    print("   ❌ Invalid PDF format")
                    return False, {}
                
                content_length = len(pdf_response.content)
                print(f"   PDF Size: {content_length} bytes")
                
                # CRITICAL VERIFICATION: Check for signature image errors
                try:
                    pdf_content_str = pdf_response.content.decode('latin-1', errors='ignore')
                    
                    # MAIN TEST: Check for '[Signature Image Error]' messages
                    if '[Signature Image Error]' in pdf_content_str:
                        print("   ❌ CRITICAL FAILURE: '[Signature Image Error]' found in PDF!")
                        print("   🚨 The signature processing fix did NOT work correctly")
                        return False, {"error": "Signature Image Error found in PDF"}
                    else:
                        print("   🎉 CRITICAL SUCCESS: NO '[Signature Image Error]' messages in PDF")
                        print("   🎉 Signature processing fix is working correctly")
                    
                    # Check for signature placeholders (should be replaced)
                    placeholder_issues = []
                    if '[First Party Signature Placeholder]' in pdf_content_str:
                        placeholder_issues.append("Original first party placeholder found")
                    if '[First Party Signature Uploaded]' in pdf_content_str:
                        placeholder_issues.append("Uploaded first party placeholder found")
                    if '[Second Party Signature Placeholder]' in pdf_content_str:
                        placeholder_issues.append("Original second party placeholder found")
                    if '[Second Party Signature Uploaded]' in pdf_content_str:
                        placeholder_issues.append("Uploaded second party placeholder found")
                    
                    if not placeholder_issues:
                        print("   ✅ All signature placeholders correctly replaced")
                    else:
                        print("   ❌ Signature placeholder issues:")
                        for issue in placeholder_issues:
                            print(f"      - {issue}")
                    
                    # Check for signature images in PDF
                    image_indicators = ['Image', '/Image', 'PNG', 'IHDR', 'ImageReader']
                    found_images = [ind for ind in image_indicators if ind in pdf_content_str]
                    
                    if found_images:
                        print(f"   ✅ Signature images embedded in PDF: {found_images}")
                    else:
                        print("   ⚠️  Could not detect signature images in PDF structure")
                    
                    # Check for signature section
                    signature_sections = ['SIGNATURES', 'FIRST PARTY', 'SECOND PARTY']
                    found_sections = [sec for sec in signature_sections if sec in pdf_content_str]
                    
                    if found_sections:
                        print(f"   ✅ Signature sections found: {found_sections}")
                    else:
                        print("   ❌ Signature sections missing from PDF")
                    
                    return True, {
                        "pdf_size": content_length,
                        "no_signature_errors": '[Signature Image Error]' not in pdf_content_str,
                        "placeholders_replaced": len(placeholder_issues) == 0,
                        "images_embedded": len(found_images) > 0,
                        "signature_sections": found_sections
                    }
                    
                except Exception as e:
                    print(f"   ❌ Could not analyze PDF content: {str(e)}")
                    return False, {"error": f"PDF analysis failed: {str(e)}"}
            else:
                print(f"❌ PDF download failed - Status: {pdf_response.status_code}")
                return False, {"error": f"PDF download failed with status {pdf_response.status_code}"}
                
        except Exception as e:
            print(f"❌ PDF generation test failed: {str(e)}")
            return False, {"error": str(e)}

    def test_edited_pdf_with_signatures(self):
        """Test edited PDF generation with signatures"""
        if not self.contract_id:
            print("⚠️  No contract ID available for edited PDF test")
            return False, {}
        
        # Get the contract with signatures
        try:
            contract_response = requests.get(f"{self.api_url}/contracts/{self.contract_id}")
            if contract_response.status_code != 200:
                print("❌ Could not retrieve contract for edited PDF test")
                return False, {}
            
            contract = contract_response.json()
            
            # Modify the contract content
            contract['content'] = contract['content'].replace(
                "Testing comprehensive signature functionality",
                "EDITED: Testing comprehensive signature functionality with modified content"
            )
            
            # Test edited PDF generation
            edited_pdf_data = {"contract": contract}
            edited_pdf_url = f"{self.api_url}/contracts/download-pdf-edited"
            
            self.tests_run += 1
            print(f"\n🔍 Testing Edited PDF Generation with Signatures...")
            
            response = requests.post(
                edited_pdf_url,
                json=edited_pdf_data,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                self.tests_passed += 1
                print(f"✅ Edited PDF generation successful")
                
                # Verify PDF format
                if not response.content.startswith(b'%PDF'):
                    print("   ❌ Invalid PDF format")
                    return False, {}
                
                content_length = len(response.content)
                print(f"   PDF Size: {content_length} bytes")
                
                # Check for signature errors in edited PDF
                try:
                    pdf_content_str = response.content.decode('latin-1', errors='ignore')
                    
                    if '[Signature Image Error]' in pdf_content_str:
                        print("   ❌ CRITICAL FAILURE: '[Signature Image Error]' found in edited PDF!")
                        return False, {"error": "Signature Image Error in edited PDF"}
                    else:
                        print("   🎉 SUCCESS: NO '[Signature Image Error]' in edited PDF")
                    
                    # Verify edited content
                    if 'EDITED:' in pdf_content_str:
                        print("   ✅ Edited content confirmed in PDF")
                    else:
                        print("   ⚠️  Could not verify edited content in PDF")
                    
                    return True, {
                        "pdf_size": content_length,
                        "no_signature_errors": '[Signature Image Error]' not in pdf_content_str,
                        "edited_content_present": 'EDITED:' in pdf_content_str
                    }
                    
                except Exception as e:
                    print(f"   ❌ Could not analyze edited PDF: {str(e)}")
                    return False, {"error": f"Edited PDF analysis failed: {str(e)}"}
            else:
                print(f"❌ Edited PDF generation failed - Status: {response.status_code}")
                return False, {"error": f"Edited PDF failed with status {response.status_code}"}
                
        except Exception as e:
            print(f"❌ Edited PDF test failed: {str(e)}")
            return False, {"error": str(e)}

    def run_comprehensive_signature_tests(self):
        """Run all signature functionality tests"""
        print("🔥 COMPREHENSIVE SIGNATURE FUNCTIONALITY TESTING")
        print("=" * 60)
        print("Testing Focus: Real signature images with PDF generation")
        print("Critical Objective: NO '[Signature Image Error]' messages in PDFs")
        print("=" * 60)
        
        results = {}
        
        # Step 1: Generate test contract
        print("\n📋 STEP 1: Contract Generation")
        success, contract = self.generate_test_contract()
        results['contract_generation'] = success
        
        if not success:
            print("❌ Cannot proceed without contract - stopping tests")
            return results
        
        # Step 2: Test signature upload and storage
        print("\n🖼️  STEP 2: Signature Upload & Storage")
        success, upload_result = self.test_signature_upload_storage()
        results['signature_upload'] = success
        
        if not success:
            print("❌ Signature upload failed - continuing with remaining tests")
        
        # Step 3: Test signature retrieval
        print("\n📥 STEP 3: Signature Retrieval")
        success, retrieval_result = self.test_signature_retrieval()
        results['signature_retrieval'] = success
        
        # Step 4: Test PDF generation with signatures (CRITICAL)
        print("\n📄 STEP 4: PDF Generation with Signatures (CRITICAL)")
        success, pdf_result = self.test_pdf_generation_with_signatures()
        results['pdf_with_signatures'] = success
        results['pdf_details'] = pdf_result
        
        # Step 5: Test edited PDF generation with signatures
        print("\n📝 STEP 5: Edited PDF Generation with Signatures")
        success, edited_pdf_result = self.test_edited_pdf_with_signatures()
        results['edited_pdf_with_signatures'] = success
        results['edited_pdf_details'] = edited_pdf_result
        
        return results

    def print_summary(self, results):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("🏁 SIGNATURE FUNCTIONALITY TEST SUMMARY")
        print("=" * 60)
        
        print(f"📊 Tests Run: {self.tests_run}")
        print(f"✅ Tests Passed: {self.tests_passed}")
        print(f"❌ Tests Failed: {self.tests_run - self.tests_passed}")
        print(f"📈 Success Rate: {(self.tests_passed/self.tests_run*100):.1f}%")
        
        print("\n🔍 DETAILED RESULTS:")
        
        # Contract Generation
        status = "✅ PASS" if results.get('contract_generation') else "❌ FAIL"
        print(f"   Contract Generation: {status}")
        
        # Signature Upload
        status = "✅ PASS" if results.get('signature_upload') else "❌ FAIL"
        print(f"   Signature Upload: {status}")
        
        # Signature Retrieval
        status = "✅ PASS" if results.get('signature_retrieval') else "❌ FAIL"
        print(f"   Signature Retrieval: {status}")
        
        # PDF with Signatures (CRITICAL)
        pdf_success = results.get('pdf_with_signatures', False)
        pdf_details = results.get('pdf_details', {})
        no_errors = pdf_details.get('no_signature_errors', False)
        
        if pdf_success and no_errors:
            print(f"   🎉 PDF with Signatures: ✅ CRITICAL SUCCESS")
            print(f"      - NO '[Signature Image Error]' messages found")
            print(f"      - Signature processing fix WORKING correctly")
        else:
            print(f"   🚨 PDF with Signatures: ❌ CRITICAL FAILURE")
            if not no_errors:
                print(f"      - '[Signature Image Error]' messages found in PDF")
                print(f"      - Signature processing fix NOT working")
        
        # Edited PDF with Signatures
        edited_success = results.get('edited_pdf_with_signatures', False)
        edited_details = results.get('edited_pdf_details', {})
        edited_no_errors = edited_details.get('no_signature_errors', False)
        
        if edited_success and edited_no_errors:
            print(f"   ✅ Edited PDF with Signatures: PASS")
        else:
            print(f"   ❌ Edited PDF with Signatures: FAIL")
        
        print("\n🎯 CRITICAL ASSESSMENT:")
        if pdf_success and no_errors and edited_success and edited_no_errors:
            print("   🎉 ALL SIGNATURE FUNCTIONALITY TESTS PASSED")
            print("   🎉 Digital signature implementation is WORKING CORRECTLY")
            print("   🎉 NO '[Signature Image Error]' messages in any PDFs")
        else:
            print("   🚨 SIGNATURE FUNCTIONALITY HAS ISSUES")
            if not (pdf_success and no_errors):
                print("   🚨 CRITICAL: Original PDF generation with signatures FAILED")
            if not (edited_success and edited_no_errors):
                print("   🚨 CRITICAL: Edited PDF generation with signatures FAILED")
        
        print("=" * 60)

def main():
    """Main test execution"""
    tester = SignatureFocusedTester()
    
    try:
        results = tester.run_comprehensive_signature_tests()
        tester.print_summary(results)
        
        # Return appropriate exit code
        critical_success = (
            results.get('pdf_with_signatures', False) and 
            results.get('pdf_details', {}).get('no_signature_errors', False) and
            results.get('edited_pdf_with_signatures', False) and
            results.get('edited_pdf_details', {}).get('no_signature_errors', False)
        )
        
        if critical_success:
            print("\n🎉 SIGNATURE FUNCTIONALITY: ALL CRITICAL TESTS PASSED")
            sys.exit(0)
        else:
            print("\n🚨 SIGNATURE FUNCTIONALITY: CRITICAL TESTS FAILED")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Test execution failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()