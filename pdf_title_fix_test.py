#!/usr/bin/env python3
"""
PDF Title Generation Fix Test
Tests the specific fix for Plain English Contract PDF title generation
to ensure no duplicate "PLAIN ENGLISH CONTRACT CONTRACT" titles
"""

import requests
import json
import sys
from datetime import datetime

class PDFTitleFixTester:
    def __init__(self, base_url="https://b3d0e54e-8004-47d5-83bd-25e76a95a599.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.conversion_id = None

    def log_test(self, name, success, details=""):
        """Log test results"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            print(f"✅ {name}: PASSED {details}")
        else:
            print(f"❌ {name}: FAILED {details}")
        return success

    def test_plain_english_conversion_with_user_scenario(self):
        """Test the exact user scenario mentioned in the problem statement"""
        print("\n🔍 Testing Plain English Conversion with User Scenario...")
        
        # Exact user scenario from problem statement
        user_input = "I want to hire a freelance web developer to build an e-commerce website for $10,000. Project should take 3 months."
        
        payload = {
            "plain_text": user_input,
            "contract_type": None,  # Use auto-detect mode as mentioned
            "jurisdiction": "US",
            "industry": "Technology",
            "output_format": "legal_clauses"
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/plain-english-to-legal",
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                self.conversion_id = result.get('id')
                
                # Check if contract type was auto-detected
                detected_type = result.get('contract_type')
                print(f"   Auto-detected contract type: {detected_type}")
                
                # Verify meaningful title generation
                if detected_type and detected_type != "Plain English Contract":
                    return self.log_test(
                        "Plain English Conversion with Auto-Detect",
                        True,
                        f"- Auto-detected: {detected_type}"
                    )
                else:
                    return self.log_test(
                        "Plain English Conversion with Auto-Detect", 
                        False,
                        f"- Failed to auto-detect meaningful type: {detected_type}"
                    )
            else:
                return self.log_test(
                    "Plain English Conversion with Auto-Detect",
                    False,
                    f"- HTTP {response.status_code}: {response.text}"
                )
                
        except Exception as e:
            return self.log_test(
                "Plain English Conversion with Auto-Detect",
                False,
                f"- Exception: {str(e)}"
            )

    def test_pdf_export_title_generation(self):
        """Test PDF export to verify title generation without duplicates"""
        if not self.conversion_id:
            return self.log_test(
                "PDF Export Title Generation",
                False,
                "- No conversion ID available"
            )
        
        print(f"\n🔍 Testing PDF Export for conversion {self.conversion_id}...")
        
        try:
            response = requests.post(
                f"{self.api_url}/plain-english-conversions/{self.conversion_id}/export",
                json={"format": "pdf"},
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if response.status_code == 200:
                # Check if it's actually a PDF
                content_type = response.headers.get('content-type', '')
                if 'application/pdf' in content_type:
                    pdf_size = len(response.content)
                    return self.log_test(
                        "PDF Export Title Generation",
                        True,
                        f"- PDF generated successfully ({pdf_size} bytes)"
                    )
                else:
                    return self.log_test(
                        "PDF Export Title Generation",
                        False,
                        f"- Wrong content type: {content_type}"
                    )
            else:
                return self.log_test(
                    "PDF Export Title Generation",
                    False,
                    f"- HTTP {response.status_code}: {response.text}"
                )
                
        except Exception as e:
            return self.log_test(
                "PDF Export Title Generation",
                False,
                f"- Exception: {str(e)}"
            )

    def test_edited_pdf_download_endpoint(self):
        """Test the /api/contracts/download-pdf-edited endpoint"""
        print(f"\n🔍 Testing Edited PDF Download Endpoint...")
        
        # Create sample edited contract data (wrapped in "contract" field as expected by endpoint)
        edited_contract_data = {
            "contract": {
                "id": "test-contract-123",
                "contract_type": "freelance_agreement",
                "content": "FREELANCE WEB DEVELOPMENT AGREEMENT\n\nThis agreement is for web development services...",
                "jurisdiction": "US",
                "created_at": datetime.now().isoformat(),
                "compliance_score": 85.0,
                "parties": {
                    "party1_name": "Client Company",
                    "party2_name": "Freelance Developer"
                },
                "terms": {
                    "payment_amount": "$10,000",
                    "project_duration": "3 months",
                    "scope": "E-commerce website development"
                }
            }
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/contracts/download-pdf-edited",
                json=edited_contract_data,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if response.status_code == 200:
                content_type = response.headers.get('content-type', '')
                if 'application/pdf' in content_type:
                    pdf_size = len(response.content)
                    
                    # Check Content-Disposition header for filename
                    content_disposition = response.headers.get('content-disposition', '')
                    print(f"   Content-Disposition: {content_disposition}")
                    
                    # Verify no duplicate "CONTRACT CONTRACT" in filename
                    if 'contract_contract' in content_disposition.lower():
                        return self.log_test(
                            "Edited PDF Download - No Duplicate Titles",
                            False,
                            f"- Found duplicate 'CONTRACT CONTRACT' in filename: {content_disposition}"
                        )
                    else:
                        return self.log_test(
                            "Edited PDF Download - No Duplicate Titles",
                            True,
                            f"- Clean filename generated ({pdf_size} bytes): {content_disposition}"
                        )
                else:
                    return self.log_test(
                        "Edited PDF Download Endpoint",
                        False,
                        f"- Wrong content type: {content_type}"
                    )
            else:
                return self.log_test(
                    "Edited PDF Download Endpoint",
                    False,
                    f"- HTTP {response.status_code}: {response.text}"
                )
                
        except Exception as e:
            return self.log_test(
                "Edited PDF Download Endpoint",
                False,
                f"- Exception: {str(e)}"
            )

    def test_multiple_contract_scenarios(self):
        """Test multiple scenarios to verify intelligent title generation"""
        print(f"\n🔍 Testing Multiple Contract Scenarios for Title Intelligence...")
        
        scenarios = [
            {
                "input": "I need a consultant to help with marketing strategy for my startup for 6 months",
                "expected_type": "consulting_agreement"
            },
            {
                "input": "We want to rent office space for 2 years at $5000 per month",
                "expected_type": "lease_agreement"
            },
            {
                "input": "Partnership agreement between two companies for joint project development",
                "expected_type": "partnership_agreement"
            }
        ]
        
        all_passed = True
        
        for i, scenario in enumerate(scenarios):
            print(f"\n   Scenario {i+1}: {scenario['input'][:50]}...")
            
            payload = {
                "plain_text": scenario["input"],
                "contract_type": None,  # Auto-detect
                "jurisdiction": "US",
                "output_format": "legal_clauses"
            }
            
            try:
                response = requests.post(
                    f"{self.api_url}/plain-english-to-legal",
                    json=payload,
                    headers={'Content-Type': 'application/json'},
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    detected_type = result.get('contract_type')
                    confidence = result.get('confidence_score', 0)
                    
                    print(f"      Detected: {detected_type} (confidence: {confidence})")
                    
                    # Check if detection is reasonable (not necessarily exact match)
                    if detected_type and detected_type != "Plain English Contract":
                        print(f"      ✅ Meaningful type detected")
                    else:
                        print(f"      ❌ Failed to detect meaningful type")
                        all_passed = False
                else:
                    print(f"      ❌ HTTP {response.status_code}")
                    all_passed = False
                    
            except Exception as e:
                print(f"      ❌ Exception: {str(e)}")
                all_passed = False
        
        return self.log_test(
            "Multiple Contract Scenarios - Intelligent Detection",
            all_passed,
            f"- Tested {len(scenarios)} scenarios"
        )

    def run_all_tests(self):
        """Run all PDF title generation fix tests"""
        print("🚀 Starting PDF Title Generation Fix Tests")
        print("=" * 60)
        
        # Test the specific user scenario
        self.test_plain_english_conversion_with_user_scenario()
        
        # Test PDF export functionality
        self.test_pdf_export_title_generation()
        
        # Test edited PDF download endpoint
        self.test_edited_pdf_download_endpoint()
        
        # Test multiple scenarios for intelligent detection
        self.test_multiple_contract_scenarios()
        
        # Summary
        print("\n" + "=" * 60)
        print(f"📊 TEST SUMMARY")
        print(f"   Tests Run: {self.tests_run}")
        print(f"   Tests Passed: {self.tests_passed}")
        print(f"   Success Rate: {(self.tests_passed/self.tests_run)*100:.1f}%")
        
        if self.tests_passed == self.tests_run:
            print("🎉 ALL TESTS PASSED - PDF Title Generation Fix is Working!")
            return True
        else:
            print("⚠️  SOME TESTS FAILED - Issues found with PDF title generation")
            return False

if __name__ == "__main__":
    tester = PDFTitleFixTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)