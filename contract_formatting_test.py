#!/usr/bin/env python3
"""
Test contract formatting across all contract types
"""

import requests
import sys
import re

class ContractFormattingTester:
    def __init__(self, base_url="https://8fa35272-f813-4f54-99e2-91d7f9fb1b50.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0

    def log_test(self, name, success, details=""):
        """Log test results"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            print(f"✅ {name}")
        else:
            print(f"❌ {name}")
        if details:
            print(f"   {details}")

    def test_contract_type(self, contract_type, contract_data):
        """Test formatting for a specific contract type"""
        print(f"\n🔍 Testing {contract_type.upper()} Contract Formatting...")
        
        try:
            url = f"{self.api_url}/generate-contract"
            response = requests.post(url, json=contract_data, timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                contract = data.get('contract', {})
                content = contract.get('content', '')
                contract_id = contract.get('id')
                
                self.log_test(f"{contract_type} Generation", True, f"Contract ID: {contract_id}")
                
                # Test formatting requirements
                self.analyze_contract_formatting(content, contract_type)
                
                # Test PDF generation
                self.test_pdf_for_contract(contract_id, contract_type)
                
                return True
            else:
                self.log_test(f"{contract_type} Generation", False, f"Status: {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test(f"{contract_type} Generation", False, f"Exception: {str(e)}")
            return False

    def analyze_contract_formatting(self, content, contract_type):
        """Analyze contract content formatting"""
        print(f"   📝 Analyzing {contract_type} formatting...")
        
        # Check 1: No asterisk symbols (except in **bold** patterns)
        asterisk_count = content.count('*')
        bold_patterns = re.findall(r'\*\*[^*]+\*\*', content)
        expected_asterisks = len(bold_patterns) * 4  # Each **bold** has 4 asterisks
        
        print(f"      Total asterisks: {asterisk_count}")
        print(f"      Bold patterns: {len(bold_patterns)}")
        print(f"      Expected asterisks (from bold): {expected_asterisks}")
        
        if asterisk_count == expected_asterisks:
            self.log_test(f"{contract_type} Asterisk Check", True, "Only asterisks from **bold** patterns found")
        else:
            self.log_test(f"{contract_type} Asterisk Check", False, f"Unexpected asterisks found")
        
        # Check 2: Date of Execution placeholder
        if '[Date of Execution]' in content or 'Date of Execution' in content:
            self.log_test(f"{contract_type} Date Placeholder", True, "Date of Execution placeholder found")
        else:
            self.log_test(f"{contract_type} Date Placeholder", False, "Missing Date of Execution placeholder")
        
        # Check 3: Bold formatting patterns
        if len(bold_patterns) > 0:
            self.log_test(f"{contract_type} Bold Formatting", True, f"Found {len(bold_patterns)} bold patterns")
            for pattern in bold_patterns[:3]:
                print(f"         - {pattern}")
        else:
            self.log_test(f"{contract_type} Bold Formatting", False, "No bold formatting patterns found")

    def test_pdf_for_contract(self, contract_id, contract_type):
        """Test PDF generation for the contract"""
        if not contract_id:
            return
        
        try:
            url = f"{self.api_url}/contracts/{contract_id}/download-pdf"
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200 and response.content.startswith(b'%PDF'):
                # Quick check for asterisks in PDF
                pdf_text = response.content.decode('latin-1', errors='ignore')
                text_lines = [line for line in pdf_text.split('\n') 
                             if not any(keyword in line for keyword in ['obj', 'endobj', 'stream', 'endstream', '/Type', '/Font'])]
                extracted_text = '\n'.join(text_lines)
                asterisk_count = extracted_text.count('*')
                
                if asterisk_count == 0:
                    self.log_test(f"{contract_type} PDF Asterisk Check", True, "No asterisks in PDF content")
                else:
                    self.log_test(f"{contract_type} PDF Asterisk Check", False, f"Found {asterisk_count} asterisks in PDF")
            else:
                self.log_test(f"{contract_type} PDF Generation", False, f"PDF generation failed: {response.status_code}")
                
        except Exception as e:
            self.log_test(f"{contract_type} PDF Test", False, f"Exception: {str(e)}")

    def run_all_contract_tests(self):
        """Test all contract types"""
        print("🚀 Testing Contract Formatting Across All Types")
        print("=" * 60)
        
        # Test data for each contract type
        contract_tests = {
            "NDA": {
                "contract_type": "NDA",
                "jurisdiction": "US",
                "parties": {
                    "party1_name": "Acme Corporation",
                    "party1_type": "corporation",
                    "party2_name": "John Smith",
                    "party2_type": "individual"
                },
                "terms": {
                    "purpose": "Sharing confidential business information for potential partnership",
                    "duration": "2_years"
                },
                "special_clauses": ["Non-compete clause"]
            },
            "freelance_agreement": {
                "contract_type": "freelance_agreement",
                "jurisdiction": "US",
                "parties": {
                    "party1_name": "Creative Agency LLC",
                    "party1_type": "llc",
                    "party2_name": "Jane Doe",
                    "party2_type": "individual"
                },
                "terms": {
                    "scope": "Website design and development",
                    "payment_amount": "$5,000",
                    "payment_terms": "milestone"
                },
                "special_clauses": ["Revision limits"]
            },
            "partnership_agreement": {
                "contract_type": "partnership_agreement",
                "jurisdiction": "US",
                "parties": {
                    "party1_name": "Tech Innovations Inc",
                    "party1_type": "corporation",
                    "party2_name": "Digital Solutions LLC",
                    "party2_type": "llc"
                },
                "terms": {
                    "business_purpose": "Joint venture for software development",
                    "profit_split": "50/50",
                    "capital_contribution": "$25,000 each"
                },
                "special_clauses": ["IP sharing agreement"]
            }
        }
        
        # Test each contract type
        for contract_type, test_data in contract_tests.items():
            self.test_contract_type(contract_type, test_data)
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 FORMATTING TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {self.tests_run}")
        print(f"Passed: {self.tests_passed}")
        print(f"Failed: {self.tests_run - self.tests_passed}")
        print(f"Success Rate: {(self.tests_passed/self.tests_run)*100:.1f}%")
        
        return self.tests_passed == self.tests_run

def main():
    tester = ContractFormattingTester()
    success = tester.run_all_contract_tests()
    
    if success:
        print("\n🎉 All contract formatting tests passed!")
        return 0
    else:
        print("\n❌ Some formatting tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())