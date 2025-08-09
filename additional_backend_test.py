#!/usr/bin/env python3
"""
Additional backend API testing for LegalMate
Testing PDF generation and contract formatting
"""

import requests
import sys
import json
from datetime import datetime

class AdditionalBackendTester:
    def __init__(self, base_url="https://9c72c586-5126-47dc-85cc-7afca9190d08.preview.emergentagent.com"):
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

    def test_basic_endpoints(self):
        """Test basic API endpoints"""
        print("🔧 TESTING BASIC API ENDPOINTS")
        print("=" * 50)
        
        results = {}
        
        # Test root endpoint
        success, response = self.run_test("Root API Endpoint", "GET", "", 200)
        results['root'] = success
        
        # Test contract types
        success, response = self.run_test("Contract Types", "GET", "contract-types", 200)
        results['contract_types'] = success
        if success and 'types' in response:
            print(f"   Found {len(response['types'])} contract types")
        
        # Test jurisdictions
        success, response = self.run_test("Jurisdictions", "GET", "jurisdictions", 200)
        results['jurisdictions'] = success
        if success and 'jurisdictions' in response:
            print(f"   Found {len(response['jurisdictions'])} jurisdictions")
        
        # Test contracts list
        success, response = self.run_test("Contracts List", "GET", "contracts", 200)
        results['contracts_list'] = success
        if success:
            print(f"   Retrieved contracts list")
        
        return results

    def test_contract_generation_and_formatting(self):
        """Test contract generation with formatting requirements"""
        print("\n📝 TESTING CONTRACT GENERATION & FORMATTING")
        print("=" * 50)
        
        results = {}
        
        # Test NDA generation
        nda_data = {
            "contract_type": "NDA",
            "jurisdiction": "US",
            "parties": {
                "party1_name": "Format Test Corp",
                "party1_type": "corporation",
                "party2_name": "Contract Validator",
                "party2_type": "individual"
            },
            "terms": {
                "purpose": "Testing contract formatting requirements and PDF generation",
                "duration": "2_years"
            },
            "special_clauses": ["Formatting verification clause"]
        }
        
        success, response = self.run_test(
            "NDA Contract Generation",
            "POST",
            "generate-contract",
            200,
            nda_data,
            timeout=60
        )
        
        results['nda_generation'] = success
        
        if success and 'contract' in response:
            contract = response['contract']
            self.contract_id = contract.get('id')
            content = contract.get('content', '')
            
            print(f"   Contract ID: {self.contract_id}")
            print(f"   Compliance Score: {contract.get('compliance_score', 'N/A')}%")
            print(f"   Content Length: {len(content)} characters")
            
            # Check formatting requirements
            formatting_issues = []
            
            # Check for asterisk symbols (should be none)
            asterisk_count = content.count('*')
            if asterisk_count > 0:
                # Count **bold** patterns
                import re
                bold_patterns = re.findall(r'\*\*[^*]+\*\*', content)
                expected_asterisks = len(bold_patterns) * 4  # Each **text** has 4 asterisks
                
                if asterisk_count == expected_asterisks:
                    print(f"   ✅ Asterisks only in **bold** patterns: {len(bold_patterns)} patterns")
                else:
                    formatting_issues.append(f"Unexpected asterisks: {asterisk_count} total, {expected_asterisks} expected")
            else:
                print(f"   ✅ No asterisk symbols found")
            
            # Check for Date of Execution placeholder
            if '[Date of Execution]' in content or 'Date of Execution' in content:
                print(f"   ✅ Date of Execution placeholder found")
            else:
                formatting_issues.append("Missing Date of Execution placeholder")
            
            # Check for signature placeholders
            if '[First Party Signature Placeholder]' in content:
                print(f"   ✅ First party signature placeholder found")
            else:
                formatting_issues.append("Missing first party signature placeholder")
            
            if '[Second Party Signature Placeholder]' in content:
                print(f"   ✅ Second party signature placeholder found")
            else:
                formatting_issues.append("Missing second party signature placeholder")
            
            if formatting_issues:
                print(f"   ❌ Formatting issues:")
                for issue in formatting_issues:
                    print(f"      - {issue}")
            else:
                print(f"   ✅ All formatting requirements met")
            
            results['formatting_check'] = len(formatting_issues) == 0
        
        return results

    def test_pdf_generation_functionality(self):
        """Test PDF generation endpoints"""
        print("\n📄 TESTING PDF GENERATION FUNCTIONALITY")
        print("=" * 50)
        
        results = {}
        
        if not self.contract_id:
            print("⚠️  No contract ID available - skipping PDF tests")
            return {'pdf_original': False, 'pdf_edited': False}
        
        # Test original PDF download
        pdf_url = f"{self.api_url}/contracts/{self.contract_id}/download-pdf"
        
        self.tests_run += 1
        print(f"\n🔍 Testing Original PDF Download...")
        print(f"   URL: {pdf_url}")
        
        try:
            pdf_response = requests.get(pdf_url, timeout=30)
            print(f"   Status: {pdf_response.status_code}")
            
            if pdf_response.status_code == 200:
                self.tests_passed += 1
                print(f"✅ Original PDF download successful")
                
                # Check PDF properties
                content_type = pdf_response.headers.get('content-type', '')
                content_disposition = pdf_response.headers.get('content-disposition', '')
                content_length = len(pdf_response.content)
                
                print(f"   Content-Type: {content_type}")
                print(f"   Content-Disposition: {content_disposition}")
                print(f"   PDF Size: {content_length} bytes")
                
                # Verify PDF format
                if pdf_response.content.startswith(b'%PDF'):
                    print("   ✅ Valid PDF format")
                else:
                    print("   ❌ Invalid PDF format")
                
                # Check for proper headers
                if 'application/pdf' in content_type:
                    print("   ✅ Correct PDF content type")
                else:
                    print("   ❌ Incorrect content type")
                
                if 'attachment' in content_disposition:
                    print("   ✅ Proper download headers")
                else:
                    print("   ❌ Missing download headers")
                
                results['pdf_original'] = True
                
                # Test bold formatting in PDF
                try:
                    pdf_content_str = pdf_response.content.decode('latin-1', errors='ignore')
                    
                    # Check for asterisks in PDF (should be none)
                    asterisk_in_pdf = pdf_content_str.count('*')
                    if asterisk_in_pdf == 0:
                        print("   ✅ No asterisk symbols in PDF - bold formatting working")
                    else:
                        print(f"   ❌ Found {asterisk_in_pdf} asterisk symbols in PDF")
                    
                    # Look for evidence of bold formatting
                    if '<b>' in pdf_content_str or 'Bold' in pdf_content_str:
                        print("   ✅ Evidence of bold formatting in PDF")
                    else:
                        print("   ⚠️  Limited evidence of bold formatting")
                    
                except Exception as e:
                    print(f"   ⚠️  Could not analyze PDF content: {str(e)}")
            else:
                print(f"❌ Original PDF download failed")
                results['pdf_original'] = False
                
        except Exception as e:
            print(f"❌ Original PDF test failed: {str(e)}")
            results['pdf_original'] = False
        
        # Test edited PDF generation
        try:
            # Get the contract
            contract_response = requests.get(f"{self.api_url}/contracts/{self.contract_id}")
            if contract_response.status_code == 200:
                contract = contract_response.json()
                
                # Modify content
                contract['content'] = contract['content'].replace(
                    "Testing contract formatting requirements",
                    "EDITED: Testing contract formatting requirements with modified content"
                )
                
                # Test edited PDF
                edited_pdf_data = {"contract": contract}
                edited_pdf_url = f"{self.api_url}/contracts/download-pdf-edited"
                
                self.tests_run += 1
                print(f"\n🔍 Testing Edited PDF Generation...")
                
                edited_response = requests.post(
                    edited_pdf_url,
                    json=edited_pdf_data,
                    headers={'Content-Type': 'application/json'},
                    timeout=30
                )
                
                print(f"   Status: {edited_response.status_code}")
                
                if edited_response.status_code == 200:
                    self.tests_passed += 1
                    print(f"✅ Edited PDF generation successful")
                    
                    content_length = len(edited_response.content)
                    print(f"   PDF Size: {content_length} bytes")
                    
                    # Verify PDF format
                    if edited_response.content.startswith(b'%PDF'):
                        print("   ✅ Valid PDF format")
                    else:
                        print("   ❌ Invalid PDF format")
                    
                    # Check filename includes 'edited'
                    content_disposition = edited_response.headers.get('content-disposition', '')
                    if '_edited.pdf' in content_disposition:
                        print("   ✅ Filename includes 'edited' indicator")
                    else:
                        print("   ⚠️  Filename doesn't include 'edited' indicator")
                    
                    results['pdf_edited'] = True
                else:
                    print(f"❌ Edited PDF generation failed")
                    results['pdf_edited'] = False
            else:
                print("❌ Could not retrieve contract for edited PDF test")
                results['pdf_edited'] = False
                
        except Exception as e:
            print(f"❌ Edited PDF test failed: {str(e)}")
            results['pdf_edited'] = False
        
        return results

    def run_comprehensive_backend_tests(self):
        """Run all backend tests"""
        print("🔧 COMPREHENSIVE BACKEND API TESTING")
        print("=" * 60)
        
        all_results = {}
        
        # Test basic endpoints
        basic_results = self.test_basic_endpoints()
        all_results.update(basic_results)
        
        # Test contract generation and formatting
        contract_results = self.test_contract_generation_and_formatting()
        all_results.update(contract_results)
        
        # Test PDF generation
        pdf_results = self.test_pdf_generation_functionality()
        all_results.update(pdf_results)
        
        return all_results

    def print_summary(self, results):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("🏁 BACKEND API TEST SUMMARY")
        print("=" * 60)
        
        print(f"📊 Tests Run: {self.tests_run}")
        print(f"✅ Tests Passed: {self.tests_passed}")
        print(f"❌ Tests Failed: {self.tests_run - self.tests_passed}")
        print(f"📈 Success Rate: {(self.tests_passed/self.tests_run*100):.1f}%")
        
        print("\n🔍 DETAILED RESULTS:")
        
        # Basic endpoints
        basic_tests = ['root', 'contract_types', 'jurisdictions', 'contracts_list']
        for test in basic_tests:
            status = "✅ PASS" if results.get(test) else "❌ FAIL"
            print(f"   {test.replace('_', ' ').title()}: {status}")
        
        # Contract generation
        status = "✅ PASS" if results.get('nda_generation') else "❌ FAIL"
        print(f"   Contract Generation: {status}")
        
        # Formatting check
        status = "✅ PASS" if results.get('formatting_check') else "❌ FAIL"
        print(f"   Contract Formatting: {status}")
        
        # PDF generation
        status = "✅ PASS" if results.get('pdf_original') else "❌ FAIL"
        print(f"   Original PDF Generation: {status}")
        
        status = "✅ PASS" if results.get('pdf_edited') else "❌ FAIL"
        print(f"   Edited PDF Generation: {status}")
        
        print("\n🎯 OVERALL ASSESSMENT:")
        critical_tests = ['nda_generation', 'formatting_check', 'pdf_original', 'pdf_edited']
        critical_passed = sum(1 for test in critical_tests if results.get(test, False))
        
        if critical_passed == len(critical_tests):
            print("   🎉 ALL CRITICAL BACKEND FUNCTIONALITY WORKING")
        else:
            print(f"   ⚠️  {critical_passed}/{len(critical_tests)} critical tests passed")
        
        print("=" * 60)

def main():
    """Main test execution"""
    tester = AdditionalBackendTester()
    
    try:
        results = tester.run_comprehensive_backend_tests()
        tester.print_summary(results)
        
        # Check critical functionality
        critical_tests = ['nda_generation', 'formatting_check', 'pdf_original', 'pdf_edited']
        critical_passed = sum(1 for test in critical_tests if results.get(test, False))
        
        if critical_passed == len(critical_tests):
            print("\n🎉 BACKEND FUNCTIONALITY: ALL CRITICAL TESTS PASSED")
            sys.exit(0)
        else:
            print(f"\n⚠️  BACKEND FUNCTIONALITY: {critical_passed}/{len(critical_tests)} CRITICAL TESTS PASSED")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Test execution failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()