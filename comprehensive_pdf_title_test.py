#!/usr/bin/env python3
"""
Comprehensive PDF Title Generation Verification Test
Specifically tests the user-reported duplicate "PLAIN ENGLISH CONTRACT CONTRACT" issue
"""

import requests
import json
import sys
from datetime import datetime

class ComprehensivePDFTitleTest:
    def __init__(self, base_url="https://legal-research-api.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0

    def log_test(self, name, success, details=""):
        """Log test results"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            print(f"✅ {name}: PASSED {details}")
        else:
            print(f"❌ {name}: FAILED {details}")
        return success

    def test_exact_user_scenario_with_filename_verification(self):
        """Test the exact user scenario and verify filename generation"""
        print("\n🔍 Testing Exact User Scenario with Filename Verification...")
        
        # Exact user input from problem statement
        user_input = "I want to hire a freelance web developer to build an e-commerce website for $10,000. Project should take 3 months."
        
        payload = {
            "plain_text": user_input,
            "contract_type": None,  # Use auto_detect mode as specified
            "jurisdiction": "US",
            "industry": "Technology",
            "output_format": "legal_clauses"
        }
        
        try:
            # Step 1: Create conversion
            response = requests.post(
                f"{self.api_url}/plain-english-to-legal",
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if response.status_code != 200:
                return self.log_test(
                    "User Scenario - Conversion Creation",
                    False,
                    f"- HTTP {response.status_code}: {response.text}"
                )
            
            result = response.json()
            conversion_id = result.get('id')
            detected_type = result.get('contract_type')
            
            print(f"   ✓ Conversion created: {conversion_id}")
            print(f"   ✓ Auto-detected type: {detected_type}")
            
            # Step 2: Export as PDF and check filename
            pdf_response = requests.post(
                f"{self.api_url}/plain-english-conversions/{conversion_id}/export",
                json={"format": "pdf"},
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if pdf_response.status_code != 200:
                return self.log_test(
                    "User Scenario - PDF Export",
                    False,
                    f"- HTTP {pdf_response.status_code}: {pdf_response.text}"
                )
            
            # Check Content-Disposition header for filename
            content_disposition = pdf_response.headers.get('content-disposition', '')
            print(f"   ✓ Content-Disposition: {content_disposition}")
            
            # Critical check: No duplicate "CONTRACT CONTRACT" in filename
            filename_lower = content_disposition.lower()
            has_duplicate_contract = 'contract_contract' in filename_lower or 'plain_english_contract_contract' in filename_lower
            
            if has_duplicate_contract:
                return self.log_test(
                    "User Scenario - No Duplicate Contract Titles",
                    False,
                    f"- Found duplicate 'CONTRACT CONTRACT' in filename: {content_disposition}"
                )
            
            # Check for meaningful filename generation
            if 'web_development' in filename_lower or 'service_agreement' in filename_lower or 'freelance' in filename_lower:
                meaningful_filename = True
                filename_type = "content-based"
            elif detected_type and detected_type.lower().replace(' ', '_') in filename_lower:
                meaningful_filename = True
                filename_type = "type-based"
            else:
                meaningful_filename = False
                filename_type = "generic"
            
            return self.log_test(
                "User Scenario - Meaningful PDF Filename Generation",
                meaningful_filename,
                f"- Filename type: {filename_type}, No duplicates: ✓, Size: {len(pdf_response.content)} bytes"
            )
            
        except Exception as e:
            return self.log_test(
                "User Scenario - Complete Test",
                False,
                f"- Exception: {str(e)}"
            )

    def test_multiple_scenarios_for_duplicate_prevention(self):
        """Test multiple scenarios to ensure no duplicate titles are generated"""
        print("\n🔍 Testing Multiple Scenarios for Duplicate Title Prevention...")
        
        test_scenarios = [
            {
                "name": "Service Agreement",
                "input": "I need someone to provide consulting services for my business",
                "expected_no_duplicates": True
            },
            {
                "name": "Employment Contract", 
                "input": "We want to hire a full-time employee with benefits and salary",
                "expected_no_duplicates": True
            },
            {
                "name": "Rental Agreement",
                "input": "I want to rent out my apartment for $2000 per month",
                "expected_no_duplicates": True
            }
        ]
        
        all_passed = True
        
        for scenario in test_scenarios:
            print(f"\n   Testing: {scenario['name']}")
            
            payload = {
                "plain_text": scenario["input"],
                "contract_type": None,  # Auto-detect
                "jurisdiction": "US",
                "output_format": "legal_clauses"
            }
            
            try:
                # Create conversion
                response = requests.post(
                    f"{self.api_url}/plain-english-to-legal",
                    json=payload,
                    headers={'Content-Type': 'application/json'},
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    conversion_id = result.get('id')
                    detected_type = result.get('contract_type')
                    
                    # Export as PDF
                    pdf_response = requests.post(
                        f"{self.api_url}/plain-english-conversions/{conversion_id}/export",
                        json={"format": "pdf"},
                        headers={'Content-Type': 'application/json'},
                        timeout=30
                    )
                    
                    if pdf_response.status_code == 200:
                        content_disposition = pdf_response.headers.get('content-disposition', '')
                        filename_lower = content_disposition.lower()
                        
                        # Check for duplicates
                        has_duplicates = (
                            'contract_contract' in filename_lower or 
                            'plain_english_contract_contract' in filename_lower or
                            'agreement_agreement' in filename_lower
                        )
                        
                        if has_duplicates:
                            print(f"      ❌ Found duplicate titles in: {content_disposition}")
                            all_passed = False
                        else:
                            print(f"      ✅ Clean filename: {content_disposition}")
                    else:
                        print(f"      ❌ PDF export failed: {pdf_response.status_code}")
                        all_passed = False
                else:
                    print(f"      ❌ Conversion failed: {response.status_code}")
                    all_passed = False
                    
            except Exception as e:
                print(f"      ❌ Exception: {str(e)}")
                all_passed = False
        
        return self.log_test(
            "Multiple Scenarios - Duplicate Prevention",
            all_passed,
            f"- Tested {len(test_scenarios)} scenarios"
        )

    def test_edited_pdf_title_generation(self):
        """Test edited PDF endpoint for title generation"""
        print("\n🔍 Testing Edited PDF Title Generation...")
        
        # Test with freelance agreement (matching user scenario)
        contract_data = {
            "contract": {
                "id": "test-freelance-web-dev",
                "contract_type": "freelance_agreement",
                "content": "FREELANCE WEB DEVELOPMENT AGREEMENT\n\nThis agreement is between Client and Developer for e-commerce website development.",
                "jurisdiction": "US",
                "created_at": datetime.now().isoformat(),
                "compliance_score": 90.0
            }
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/contracts/download-pdf-edited",
                json=contract_data,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if response.status_code == 200:
                content_disposition = response.headers.get('content-disposition', '')
                filename_lower = content_disposition.lower()
                
                # Check for duplicates
                has_duplicates = (
                    'contract_contract' in filename_lower or 
                    'plain_english_contract_contract' in filename_lower
                )
                
                if has_duplicates:
                    return self.log_test(
                        "Edited PDF - No Duplicate Titles",
                        False,
                        f"- Found duplicates in: {content_disposition}"
                    )
                else:
                    return self.log_test(
                        "Edited PDF - No Duplicate Titles",
                        True,
                        f"- Clean filename: {content_disposition}, Size: {len(response.content)} bytes"
                    )
            else:
                return self.log_test(
                    "Edited PDF - Title Generation",
                    False,
                    f"- HTTP {response.status_code}: {response.text}"
                )
                
        except Exception as e:
            return self.log_test(
                "Edited PDF - Title Generation",
                False,
                f"- Exception: {str(e)}"
            )

    def run_comprehensive_tests(self):
        """Run all comprehensive PDF title generation tests"""
        print("🚀 Starting Comprehensive PDF Title Generation Tests")
        print("=" * 70)
        print("🎯 FOCUS: Verifying fix for duplicate 'PLAIN ENGLISH CONTRACT CONTRACT' issue")
        print("=" * 70)
        
        # Test exact user scenario
        self.test_exact_user_scenario_with_filename_verification()
        
        # Test multiple scenarios for duplicate prevention
        self.test_multiple_scenarios_for_duplicate_prevention()
        
        # Test edited PDF endpoint
        self.test_edited_pdf_title_generation()
        
        # Summary
        print("\n" + "=" * 70)
        print(f"📊 COMPREHENSIVE TEST SUMMARY")
        print(f"   Tests Run: {self.tests_run}")
        print(f"   Tests Passed: {self.tests_passed}")
        print(f"   Success Rate: {(self.tests_passed/self.tests_run)*100:.1f}%")
        
        if self.tests_passed == self.tests_run:
            print("🎉 ALL COMPREHENSIVE TESTS PASSED!")
            print("✅ PDF Title Generation Fix is FULLY WORKING")
            print("✅ No duplicate 'PLAIN ENGLISH CONTRACT CONTRACT' titles found")
            print("✅ Intelligent title generation is operational")
            return True
        else:
            print("⚠️  SOME COMPREHENSIVE TESTS FAILED")
            print("❌ Issues remain with PDF title generation")
            return False

if __name__ == "__main__":
    tester = ComprehensivePDFTitleTest()
    success = tester.run_comprehensive_tests()
    sys.exit(0 if success else 1)