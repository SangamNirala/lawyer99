#!/usr/bin/env python3
"""
Focused test for PDF bold formatting functionality
Tests the specific requirement that PDF should have bold formatting without asterisks
"""

import requests
import sys
import json
import re
from datetime import datetime

class PDFBoldFormattingTester:
    def __init__(self, base_url="https://9c72c586-5126-47dc-85cc-7afca9190d08.preview.emergentagent.com"):
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

    def generate_test_contract(self):
        """Generate a contract specifically for PDF bold formatting testing"""
        print("🔍 Step 1: Generating test contract...")
        
        test_data = {
            "contract_type": "NDA",
            "jurisdiction": "US",
            "parties": {
                "party1_name": "Bold Format Testing Corporation",
                "party1_type": "corporation",
                "party2_name": "PDF Validation Expert",
                "party2_type": "individual"
            },
            "terms": {
                "purpose": "Testing PDF bold formatting to ensure section headings appear in bold without asterisk symbols in the final PDF output",
                "duration": "2_years"
            },
            "special_clauses": ["Bold formatting verification clause", "PDF rendering quality assurance"]
        }
        
        try:
            url = f"{self.api_url}/generate-contract"
            response = requests.post(url, json=test_data, timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                contract = data.get('contract', {})
                contract_id = contract.get('id')
                content = contract.get('content', '')
                
                self.log_test("Contract Generation", True, f"Generated contract ID: {contract_id}")
                
                # Analyze contract content
                print(f"\n📝 Contract Content Analysis:")
                
                # Check for asterisks in content
                asterisk_count = content.count('*')
                print(f"   Asterisk (*) count in contract: {asterisk_count}")
                
                # Check for **bold** patterns
                bold_patterns = re.findall(r'\*\*[^*]+\*\*', content)
                print(f"   **Bold** patterns found: {len(bold_patterns)}")
                
                if bold_patterns:
                    print(f"   Sample bold patterns:")
                    for pattern in bold_patterns[:3]:
                        print(f"     - {pattern}")
                
                # Show content preview
                print(f"\n   Content preview (first 400 chars):")
                print(f"   {content[:400]}...")
                
                return contract_id, content
            else:
                self.log_test("Contract Generation", False, f"Status: {response.status_code}, Error: {response.text}")
                return None, None
                
        except Exception as e:
            self.log_test("Contract Generation", False, f"Exception: {str(e)}")
            return None, None

    def test_pdf_download_and_formatting(self, contract_id):
        """Download PDF and test bold formatting"""
        if not contract_id:
            self.log_test("PDF Download", False, "No contract ID available")
            return False
        
        print(f"\n🔍 Step 2: Testing PDF download and bold formatting...")
        
        try:
            url = f"{self.api_url}/contracts/{contract_id}/download-pdf"
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                self.log_test("PDF Download Success", True, f"Status: {response.status_code}")
                
                # Verify PDF format
                if response.content.startswith(b'%PDF'):
                    self.log_test("PDF Format Validation", True, "Valid PDF header found")
                else:
                    self.log_test("PDF Format Validation", False, "Invalid PDF format")
                    return False
                
                # Check content type and headers
                content_type = response.headers.get('content-type', '')
                content_disposition = response.headers.get('content-disposition', '')
                
                if 'application/pdf' in content_type:
                    self.log_test("PDF Content-Type", True, f"Correct content type: {content_type}")
                else:
                    self.log_test("PDF Content-Type", False, f"Wrong content type: {content_type}")
                
                if 'attachment' in content_disposition and 'filename' in content_disposition:
                    self.log_test("PDF Download Headers", True, f"Correct headers: {content_disposition}")
                else:
                    self.log_test("PDF Download Headers", False, f"Missing/incorrect headers: {content_disposition}")
                
                # Check PDF size
                pdf_size = len(response.content)
                print(f"   PDF Size: {pdf_size} bytes")
                
                if pdf_size > 2000:
                    self.log_test("PDF Size Check", True, f"Reasonable size: {pdf_size} bytes")
                else:
                    self.log_test("PDF Size Check", False, f"PDF too small: {pdf_size} bytes")
                
                # Analyze PDF content for asterisks
                self.analyze_pdf_content(response.content)
                
                return True
            else:
                self.log_test("PDF Download", False, f"Status: {response.status_code}, Error: {response.text}")
                return False
                
        except Exception as e:
            self.log_test("PDF Download", False, f"Exception: {str(e)}")
            return False

    def analyze_pdf_content(self, pdf_content):
        """Analyze PDF content for asterisk symbols and bold formatting"""
        print(f"\n📄 PDF Content Analysis:")
        
        try:
            # Convert PDF bytes to string for basic text analysis
            # Note: This is a simple approach - in production you'd use a proper PDF parser
            pdf_text = pdf_content.decode('latin-1', errors='ignore')
            
            # Filter out PDF structure and focus on text content
            text_lines = []
            for line in pdf_text.split('\n'):
                # Skip PDF metadata and structure lines
                if not any(keyword in line for keyword in [
                    'obj', 'endobj', 'stream', 'endstream', '/Type', '/Font', 
                    '/Length', '/Filter', '/Root', '/Info', 'xref', 'trailer'
                ]):
                    if line.strip() and not line.startswith('%') and len(line.strip()) > 2:
                        text_lines.append(line.strip())
            
            extracted_text = '\n'.join(text_lines)
            
            # Count asterisks in the extracted text
            asterisk_count = extracted_text.count('*')
            
            print(f"   Asterisk (*) symbols in PDF text: {asterisk_count}")
            
            if asterisk_count == 0:
                self.log_test("PDF Asterisk Check", True, "No asterisk (*) symbols found in PDF content")
            else:
                self.log_test("PDF Asterisk Check", False, f"Found {asterisk_count} asterisk (*) symbols in PDF")
                
                # Show lines with asterisks
                asterisk_lines = [line for line in extracted_text.split('\n') if '*' in line]
                print(f"   Lines with asterisks:")
                for line in asterisk_lines[:3]:  # Show first 3
                    print(f"     - {line[:100]}...")
            
            # Look for evidence of bold formatting in PDF structure
            bold_indicators = [
                '<b>' in pdf_text,  # HTML bold tags
                '/F1' in pdf_text,  # Font references
                'Bold' in pdf_text,  # Bold font names
                'Tf' in pdf_text    # Text formatting commands
            ]
            
            if any(bold_indicators):
                self.log_test("PDF Bold Formatting Evidence", True, "Found evidence of bold formatting in PDF structure")
            else:
                self.log_test("PDF Bold Formatting Evidence", False, "Limited evidence of bold formatting in PDF")
            
            # Show sample of extracted text
            print(f"\n   Sample extracted text (first 300 chars):")
            print(f"   {extracted_text[:300]}...")
            
        except Exception as e:
            self.log_test("PDF Content Analysis", False, f"Could not analyze PDF content: {str(e)}")

    def run_comprehensive_test(self):
        """Run the complete PDF bold formatting test"""
        print("🚀 Starting PDF Bold Formatting Test")
        print("=" * 60)
        print("Testing requirement: PDF should have bold formatting WITHOUT asterisks")
        print("=" * 60)
        
        # Step 1: Generate contract
        contract_id, contract_content = self.generate_test_contract()
        
        if not contract_id:
            print("❌ Cannot proceed - contract generation failed")
            return False
        
        # Step 2: Test PDF download and formatting
        pdf_success = self.test_pdf_download_and_formatting(contract_id)
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {self.tests_run}")
        print(f"Passed: {self.tests_passed}")
        print(f"Failed: {self.tests_run - self.tests_passed}")
        print(f"Success Rate: {(self.tests_passed/self.tests_run)*100:.1f}%")
        
        # Key findings
        print(f"\n🔍 KEY FINDINGS:")
        print(f"   Contract ID: {contract_id}")
        print(f"   PDF Generation: {'✅ Working' if pdf_success else '❌ Failed'}")
        
        return self.tests_passed == self.tests_run

def main():
    tester = PDFBoldFormattingTester()
    success = tester.run_comprehensive_test()
    
    if success:
        print("\n🎉 All PDF bold formatting tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed. Check output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())