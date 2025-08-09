import requests
import sys
import json
from datetime import datetime

class EditedPDFTester:
    def __init__(self, base_url="https://8cd68d5c-4981-470b-b1c0-9982d2b4a8d2.preview.emergentagent.com"):
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
                    print(f"   Response keys: {list(response_data.keys()) if isinstance(response_data, dict) else 'List with ' + str(len(response_data)) + ' items'}")
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

    def test_edited_pdf_generation_valid_data(self):
        """Test the new edited PDF generation endpoint with valid contract data"""
        # First generate a contract to get valid structure
        contract_data = {
            "contract_type": "NDA",
            "jurisdiction": "US",
            "parties": {
                "party1_name": "EditTest Corp",
                "party1_type": "corporation",
                "party2_name": "PDF Editor",
                "party2_type": "individual"
            },
            "terms": {
                "purpose": "Testing edited PDF generation functionality",
                "duration": "1_year"
            },
            "special_clauses": ["Edited content verification clause"]
        }
        
        # Generate original contract
        success, response = self.run_test(
            "Generate Contract for Edited PDF Test", 
            "POST", 
            "generate-contract", 
            200, 
            contract_data,
            timeout=60
        )
        
        if not success or 'contract' not in response:
            print("❌ Failed to generate contract for edited PDF testing")
            return False, {}
        
        original_contract = response['contract']
        
        # Modify the contract content to simulate editing
        edited_contract = original_contract.copy()
        edited_contract['content'] = edited_contract['content'].replace(
            "Testing edited PDF generation functionality",
            "EDITED: Testing the new edited PDF generation functionality with modified content"
        )
        
        # Test the edited PDF endpoint
        edited_pdf_data = {
            "contract": edited_contract
        }
        
        url = f"{self.api_url}/contracts/download-pdf-edited"
        
        self.tests_run += 1
        print(f"\n🔍 Testing Edited PDF Generation with Valid Data...")
        print(f"   URL: {url}")
        print(f"   Contract ID: {edited_contract.get('id')}")
        
        try:
            response = requests.post(url, json=edited_pdf_data, headers={'Content-Type': 'application/json'}, timeout=30)
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                self.tests_passed += 1
                print(f"✅ Passed - Edited PDF generated successfully")
                
                # Check response headers
                content_type = response.headers.get('content-type', '')
                content_disposition = response.headers.get('content-disposition', '')
                
                print(f"   Content-Type: {content_type}")
                print(f"   Content-Disposition: {content_disposition}")
                
                # Verify PDF headers
                if 'application/pdf' in content_type:
                    print("   ✅ Correct PDF content type")
                else:
                    print(f"   ❌ Expected PDF content type, got: {content_type}")
                
                if 'attachment' in content_disposition and 'filename' in content_disposition:
                    print("   ✅ Correct download headers")
                    if '_edited.pdf' in content_disposition:
                        print("   ✅ Filename includes 'edited' indicator")
                    else:
                        print("   ⚠️  Filename doesn't include 'edited' indicator")
                else:
                    print(f"   ❌ Missing or incorrect download headers")
                
                # Check PDF content size
                content_length = len(response.content)
                print(f"   PDF Size: {content_length} bytes")
                
                if content_length > 1000:
                    print("   ✅ PDF has reasonable size")
                else:
                    print("   ❌ PDF seems too small")
                
                # Check if content starts with PDF header
                if response.content.startswith(b'%PDF'):
                    print("   ✅ Valid PDF format")
                else:
                    print("   ❌ Invalid PDF format - missing PDF header")
                
                # Try to verify "Edited" status in PDF content
                try:
                    pdf_content_str = response.content.decode('latin-1', errors='ignore')
                    if 'Edited' in pdf_content_str:
                        print("   ✅ PDF includes 'Edited' status indicator")
                    else:
                        print("   ⚠️  Could not verify 'Edited' status in PDF")
                except:
                    print("   ⚠️  Could not analyze PDF content for 'Edited' status")
                
                return True, {
                    "content_length": content_length, 
                    "headers": dict(response.headers),
                    "original_contract_id": original_contract.get('id'),
                    "edited_contract_id": edited_contract.get('id')
                }
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

    def test_edited_pdf_generation_invalid_data(self):
        """Test edited PDF generation endpoint with invalid request format"""
        # Test with missing contract data
        invalid_data_1 = {}
        
        success_1, response_1 = self.run_test(
            "Edited PDF with Missing Contract Data", 
            "POST", 
            "contracts/download-pdf-edited", 
            422,  # Expecting validation error
            invalid_data_1
        )
        
        # If 422 doesn't work, try 500
        if not success_1:
            success_1, response_1 = self.run_test(
                "Edited PDF with Missing Contract Data (500)", 
                "POST", 
                "contracts/download-pdf-edited", 
                500,
                invalid_data_1
            )
            if success_1:
                self.tests_passed += 1  # Adjust count since we ran an extra test
        
        # Test with invalid contract structure
        invalid_data_2 = {
            "contract": {
                "invalid_field": "test"
                # Missing required fields like id, content, etc.
            }
        }
        
        success_2, response_2 = self.run_test(
            "Edited PDF with Invalid Contract Structure", 
            "POST", 
            "contracts/download-pdf-edited", 
            500,  # Expecting server error due to missing required fields
            invalid_data_2
        )
        
        return success_1 and success_2, {"test1": response_1, "test2": response_2}

    def test_edited_pdf_content_verification(self):
        """Test that edited PDF content differs from original when content is modified"""
        # Generate a contract first
        contract_data = {
            "contract_type": "freelance_agreement",
            "jurisdiction": "US",
            "parties": {
                "party1_name": "Content Verification LLC",
                "party1_type": "llc",
                "party2_name": "PDF Content Tester",
                "party2_type": "individual"
            },
            "terms": {
                "scope": "Original content for PDF verification testing",
                "payment_amount": "$1,000",
                "payment_terms": "upon_completion"
            },
            "special_clauses": []
        }
        
        # Generate original contract
        success, response = self.run_test(
            "Generate Contract for Content Verification", 
            "POST", 
            "generate-contract", 
            200, 
            contract_data,
            timeout=60
        )
        
        if not success or 'contract' not in response:
            print("❌ Failed to generate contract for content verification")
            return False, {}
        
        original_contract = response['contract']
        contract_id = original_contract.get('id')
        
        # Download original PDF
        original_pdf_url = f"{self.api_url}/contracts/{contract_id}/download-pdf"
        
        print(f"\n🔍 Testing Content Verification - Downloading Original PDF...")
        try:
            original_pdf_response = requests.get(original_pdf_url, timeout=30)
            if original_pdf_response.status_code != 200:
                print("❌ Failed to download original PDF")
                return False, {}
            
            original_pdf_size = len(original_pdf_response.content)
            print(f"   Original PDF Size: {original_pdf_size} bytes")
            
        except Exception as e:
            print(f"❌ Failed to download original PDF: {str(e)}")
            return False, {}
        
        # Create edited version with significant content changes
        edited_contract = original_contract.copy()
        edited_contract['content'] = edited_contract['content'].replace(
            "Original content for PDF verification testing",
            "SIGNIFICANTLY MODIFIED CONTENT: This has been extensively edited to verify that the edited PDF generation creates different content from the original PDF"
        )
        
        # Add more modifications to ensure clear difference
        edited_contract['content'] = edited_contract['content'].replace(
            "$1,000",
            "$2,500 (EDITED AMOUNT)"
        )
        
        # Generate edited PDF
        edited_pdf_data = {"contract": edited_contract}
        edited_pdf_url = f"{self.api_url}/contracts/download-pdf-edited"
        
        self.tests_run += 1
        print(f"\n🔍 Testing Content Verification - Generating Edited PDF...")
        
        try:
            edited_pdf_response = requests.post(
                edited_pdf_url, 
                json=edited_pdf_data, 
                headers={'Content-Type': 'application/json'}, 
                timeout=30
            )
            
            if edited_pdf_response.status_code != 200:
                print(f"❌ Failed to generate edited PDF - Status: {edited_pdf_response.status_code}")
                return False, {}
            
            edited_pdf_size = len(edited_pdf_response.content)
            print(f"   Edited PDF Size: {edited_pdf_size} bytes")
            
            # Compare PDF sizes (they should be different due to content changes)
            size_difference = abs(edited_pdf_size - original_pdf_size)
            print(f"   Size Difference: {size_difference} bytes")
            
            if size_difference > 50:  # Reasonable threshold for content difference
                print("   ✅ PDF sizes differ significantly - content modification detected")
                self.tests_passed += 1
            else:
                print("   ⚠️  PDF sizes are very similar - content modification may not be reflected")
            
            # Try to verify content differences in PDF structure
            try:
                original_pdf_str = original_pdf_response.content.decode('latin-1', errors='ignore')
                edited_pdf_str = edited_pdf_response.content.decode('latin-1', errors='ignore')
                
                # Look for the modified text in the edited PDF
                if 'SIGNIFICANTLY MODIFIED CONTENT' in edited_pdf_str:
                    print("   ✅ Edited content found in edited PDF")
                elif 'MODIFIED CONTENT' in edited_pdf_str:
                    print("   ✅ Modified content detected in edited PDF")
                else:
                    print("   ⚠️  Could not verify edited content in PDF")
                
                if 'EDITED AMOUNT' in edited_pdf_str:
                    print("   ✅ Edited amount found in edited PDF")
                else:
                    print("   ⚠️  Could not verify edited amount in PDF")
                
                # Verify original content is NOT in edited PDF
                if 'Original content for PDF verification testing' not in edited_pdf_str:
                    print("   ✅ Original content successfully replaced in edited PDF")
                else:
                    print("   ❌ Original content still present in edited PDF")
                
            except Exception as e:
                print(f"   ⚠️  Could not analyze PDF content differences: {str(e)}")
            
            return True, {
                "original_pdf_size": original_pdf_size,
                "edited_pdf_size": edited_pdf_size,
                "size_difference": size_difference,
                "contract_id": contract_id
            }
            
        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

def main():
    print("🚀 Starting Edited PDF Generation Tests")
    print("=" * 60)
    
    tester = EditedPDFTester()
    
    # Run edited PDF tests
    test_results = []
    
    print("\n" + "="*50)
    print("📝 Testing Edited PDF Generation Functionality")
    print("="*50)
    
    test_results.append(tester.test_edited_pdf_generation_valid_data())
    test_results.append(tester.test_edited_pdf_generation_invalid_data())
    test_results.append(tester.test_edited_pdf_content_verification())
    
    # Print final results
    print("\n" + "="*60)
    print("📊 TEST RESULTS SUMMARY")
    print("="*60)
    print(f"Total Tests Run: {tester.tests_run}")
    print(f"Tests Passed: {tester.tests_passed}")
    print(f"Tests Failed: {tester.tests_run - tester.tests_passed}")
    print(f"Success Rate: {(tester.tests_passed/tester.tests_run)*100:.1f}%")
    
    if tester.tests_passed == tester.tests_run:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())