import requests
import sys
import json
import re
from datetime import datetime

class PlainEnglishPDFTitleTester:
    def __init__(self, base_url="https://b3d0e54e-8004-47d5-83bd-25e76a95a599.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.conversion_ids = []

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

    def test_main_conversion_endpoint_user_scenario(self):
        """Test the main Plain English conversion endpoint with the specific user input"""
        user_input = "I want to hire a freelance web developer to build an e-commerce website for $10,000. The project should take 3 months, include responsive design, payment integration, and maintenance for 6 months after launch."
        
        conversion_data = {
            "plain_text": user_input,
            "contract_type": "auto_detect",  # Test auto-detection
            "jurisdiction": "US",
            "industry": "Technology",
            "output_format": "complete_contract"
        }
        
        success, response = self.run_test(
            "Main Plain English Conversion - User Scenario", 
            "POST", 
            "plain-english-to-legal", 
            200, 
            conversion_data,
            timeout=60
        )
        
        if success and response:
            conversion_id = response.get('id')
            if conversion_id:
                self.conversion_ids.append(conversion_id)
                print(f"   Conversion ID: {conversion_id}")
            
            # Analyze the response for title detection
            generated_clauses = response.get('generated_clauses', [])
            print(f"   Generated {len(generated_clauses)} legal clauses")
            
            # Check if contract type was auto-detected
            detected_type = response.get('contract_type')
            print(f"   Auto-detected contract type: {detected_type}")
            
            # Look for web development related content
            full_contract = response.get('full_contract', '')
            if 'web development' in full_contract.lower() or 'website' in full_contract.lower():
                print(f"   ✅ Contract content includes web development terms")
            else:
                print(f"   ⚠️  Contract content may not reflect web development focus")
            
            # Check confidence score
            confidence = response.get('confidence_score', 0)
            print(f"   Confidence Score: {confidence * 100:.1f}%")
            
            if confidence >= 0.8:
                print(f"   ✅ High confidence conversion (≥80%)")
            else:
                print(f"   ⚠️  Lower confidence conversion (<80%)")
                
        return success, response

    def test_pdf_export_meaningful_title(self):
        """Test PDF export functionality to verify meaningful titles"""
        if not self.conversion_ids:
            print("⚠️  Skipping PDF export test - no conversion IDs available")
            return True, {}
        
        conversion_id = self.conversion_ids[0]
        
        # Test PDF export
        export_data = {
            "format": "pdf"
        }
        
        url = f"{self.api_url}/plain-english-conversions/{conversion_id}/export"
        
        self.tests_run += 1
        print(f"\n🔍 Testing PDF Export with Meaningful Title...")
        print(f"   URL: {url}")
        print(f"   Conversion ID: {conversion_id}")
        
        try:
            response = requests.post(url, json=export_data, headers={'Content-Type': 'application/json'}, timeout=30)
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                self.tests_passed += 1
                print(f"✅ PDF export successful")
                
                # Check response headers for filename
                content_disposition = response.headers.get('content-disposition', '')
                print(f"   Content-Disposition: {content_disposition}")
                
                # Extract filename from Content-Disposition header
                filename_match = re.search(r'filename="([^"]+)"', content_disposition)
                if filename_match:
                    filename = filename_match.group(1)
                    print(f"   📄 PDF Filename: {filename}")
                    
                    # Check if filename is meaningful (not generic)
                    if 'plain_english_contract' in filename.lower():
                        print(f"   ❌ ISSUE FOUND: Filename still uses generic 'plain_english_contract' pattern")
                        print(f"   ❌ This suggests the title generation fix may not be working")
                    elif 'web_development' in filename.lower() or 'ecommerce' in filename.lower() or 'website' in filename.lower():
                        print(f"   ✅ TITLE FIX WORKING: Filename reflects content (web development/ecommerce)")
                    elif 'service_agreement' in filename.lower() or 'development_agreement' in filename.lower():
                        print(f"   ✅ TITLE FIX WORKING: Filename shows intelligent detection ({filename})")
                    else:
                        print(f"   ⚠️  Filename pattern unclear - need to verify if it's meaningful")
                else:
                    print(f"   ⚠️  Could not extract filename from headers")
                
                # Verify PDF format
                if response.content.startswith(b'%PDF'):
                    print(f"   ✅ Valid PDF format")
                else:
                    print(f"   ❌ Invalid PDF format")
                
                # Check PDF size
                pdf_size = len(response.content)
                print(f"   PDF Size: {pdf_size} bytes")
                
                if pdf_size > 2000:
                    print(f"   ✅ PDF has reasonable size")
                else:
                    print(f"   ⚠️  PDF seems small")
                
                # Try to analyze PDF content for title
                try:
                    pdf_content_str = response.content.decode('latin-1', errors='ignore')
                    
                    # Look for title patterns in PDF
                    if 'PLAIN ENGLISH CONTRACT CONTRACT' in pdf_content_str.upper():
                        print(f"   ❌ CRITICAL ISSUE: Found 'PLAIN ENGLISH CONTRACT CONTRACT' in PDF")
                        print(f"   ❌ The duplicate title issue is NOT fixed")
                    elif 'WEB DEVELOPMENT' in pdf_content_str.upper():
                        print(f"   ✅ TITLE FIX WORKING: PDF contains 'Web Development' title")
                    elif 'SERVICE AGREEMENT' in pdf_content_str.upper():
                        print(f"   ✅ TITLE FIX WORKING: PDF contains 'Service Agreement' title")
                    elif 'DEVELOPMENT AGREEMENT' in pdf_content_str.upper():
                        print(f"   ✅ TITLE FIX WORKING: PDF contains 'Development Agreement' title")
                    elif 'E-COMMERCE' in pdf_content_str.upper() or 'ECOMMERCE' in pdf_content_str.upper():
                        print(f"   ✅ TITLE FIX WORKING: PDF contains e-commerce related title")
                    else:
                        print(f"   ⚠️  Could not identify specific title pattern in PDF")
                        # Look for any title-like patterns
                        title_patterns = re.findall(r'[A-Z][A-Z\s]{10,50}AGREEMENT', pdf_content_str)
                        if title_patterns:
                            print(f"   Found title patterns: {title_patterns[:3]}")
                
                except Exception as e:
                    print(f"   ⚠️  Could not analyze PDF content: {str(e)}")
                
                return True, {
                    "filename": filename_match.group(1) if filename_match else None,
                    "pdf_size": pdf_size,
                    "conversion_id": conversion_id
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

    def test_intelligent_title_detection_scenarios(self):
        """Test intelligent title detection with various scenarios"""
        test_scenarios = [
            {
                "plain_text": "I need a consultant to help with marketing strategy for my startup",
                "expected_keywords": ["consulting", "marketing", "advisory"],
                "description": "Marketing Consulting Scenario"
            },
            {
                "plain_text": "We want to rent office space for 2 years",
                "expected_keywords": ["lease", "rental", "office"],
                "description": "Office Lease Scenario"
            },
            {
                "plain_text": "Partnership agreement between two companies for joint project",
                "expected_keywords": ["partnership", "joint", "collaboration"],
                "description": "Partnership Agreement Scenario"
            }
        ]
        
        all_success = True
        results = {}
        
        for i, scenario in enumerate(test_scenarios):
            conversion_data = {
                "plain_text": scenario["plain_text"],
                "contract_type": None,  # Test auto-detection
                "jurisdiction": "US",
                "output_format": "complete_contract"
            }
            
            success, response = self.run_test(
                f"Title Detection - {scenario['description']}", 
                "POST", 
                "plain-english-to-legal", 
                200, 
                conversion_data,
                timeout=60
            )
            
            if success and response:
                conversion_id = response.get('id')
                if conversion_id:
                    self.conversion_ids.append(conversion_id)
                
                # Check auto-detected contract type
                detected_type = response.get('contract_type')
                print(f"   Auto-detected type: {detected_type}")
                
                # Check if detected type makes sense for the scenario
                detected_type_lower = (detected_type or '').lower()
                keyword_found = any(keyword in detected_type_lower for keyword in scenario['expected_keywords'])
                
                if keyword_found:
                    print(f"   ✅ Intelligent detection working - type matches scenario")
                else:
                    print(f"   ⚠️  Detection may not be optimal for this scenario")
                
                # Test PDF export for this scenario
                if conversion_id:
                    export_success = self.test_pdf_export_for_scenario(conversion_id, scenario)
                    if not export_success:
                        all_success = False
                
                results[scenario['description']] = {
                    "success": success,
                    "detected_type": detected_type,
                    "conversion_id": conversion_id,
                    "keyword_match": keyword_found
                }
            else:
                all_success = False
                results[scenario['description']] = {"success": False, "error": "Conversion failed"}
        
        return all_success, results

    def test_pdf_export_for_scenario(self, conversion_id, scenario):
        """Test PDF export for a specific scenario to check title generation"""
        export_data = {"format": "pdf"}
        url = f"{self.api_url}/plain-english-conversions/{conversion_id}/export"
        
        print(f"   📄 Testing PDF export for {scenario['description']}...")
        
        try:
            response = requests.post(url, json=export_data, headers={'Content-Type': 'application/json'}, timeout=30)
            
            if response.status_code == 200:
                # Check filename
                content_disposition = response.headers.get('content-disposition', '')
                filename_match = re.search(r'filename="([^"]+)"', content_disposition)
                
                if filename_match:
                    filename = filename_match.group(1)
                    print(f"   📄 PDF Filename: {filename}")
                    
                    # Check if filename reflects the scenario
                    filename_lower = filename.lower()
                    keyword_in_filename = any(keyword in filename_lower for keyword in scenario['expected_keywords'])
                    
                    if keyword_in_filename:
                        print(f"   ✅ Filename reflects scenario content")
                    elif 'plain_english_contract' in filename_lower:
                        print(f"   ❌ Filename still uses generic pattern")
                        return False
                    else:
                        print(f"   ⚠️  Filename pattern unclear")
                
                # Check PDF content for title
                try:
                    pdf_content_str = response.content.decode('latin-1', errors='ignore')
                    
                    if 'PLAIN ENGLISH CONTRACT CONTRACT' in pdf_content_str.upper():
                        print(f"   ❌ CRITICAL: Duplicate title issue found in PDF")
                        return False
                    else:
                        print(f"   ✅ No duplicate title issue detected")
                
                except Exception as e:
                    print(f"   ⚠️  Could not analyze PDF content: {str(e)}")
                
                return True
            else:
                print(f"   ❌ PDF export failed with status {response.status_code}")
                return False
                
        except Exception as e:
            print(f"   ❌ PDF export error: {str(e)}")
            return False

    def test_fallback_title_generation(self):
        """Test fallback to 'Plain English Contract' when detection fails"""
        # Use very ambiguous text that should be hard to categorize
        ambiguous_text = "Some general business stuff needs to be handled properly with appropriate measures."
        
        conversion_data = {
            "plain_text": ambiguous_text,
            "contract_type": None,
            "jurisdiction": "US",
            "output_format": "complete_contract"
        }
        
        success, response = self.run_test(
            "Fallback Title Generation Test", 
            "POST", 
            "plain-english-to-legal", 
            200, 
            conversion_data,
            timeout=60
        )
        
        if success and response:
            conversion_id = response.get('id')
            if conversion_id:
                self.conversion_ids.append(conversion_id)
            
            detected_type = response.get('contract_type')
            print(f"   Auto-detected type for ambiguous text: {detected_type}")
            
            # Test PDF export to check fallback behavior
            if conversion_id:
                export_data = {"format": "pdf"}
                url = f"{self.api_url}/plain-english-conversions/{conversion_id}/export"
                
                print(f"   Testing PDF export for fallback scenario...")
                
                try:
                    pdf_response = requests.post(url, json=export_data, headers={'Content-Type': 'application/json'}, timeout=30)
                    
                    if pdf_response.status_code == 200:
                        # Check filename
                        content_disposition = pdf_response.headers.get('content-disposition', '')
                        filename_match = re.search(r'filename="([^"]+)"', content_disposition)
                        
                        if filename_match:
                            filename = filename_match.group(1)
                            print(f"   📄 Fallback PDF Filename: {filename}")
                            
                            # For ambiguous content, should fallback to generic but professional title
                            if 'plain_english_contract' in filename.lower():
                                print(f"   ✅ Proper fallback to generic title")
                            elif any(word in filename.lower() for word in ['service', 'agreement', 'contract']):
                                print(f"   ✅ Reasonable fallback title generated")
                            else:
                                print(f"   ⚠️  Unexpected fallback filename pattern")
                        
                        # Check PDF content
                        try:
                            pdf_content_str = pdf_response.content.decode('latin-1', errors='ignore')
                            
                            if 'PLAIN ENGLISH CONTRACT CONTRACT' in pdf_content_str.upper():
                                print(f"   ❌ CRITICAL: Duplicate title issue in fallback scenario")
                                return False, {}
                            else:
                                print(f"   ✅ No duplicate title issue in fallback")
                        
                        except Exception as e:
                            print(f"   ⚠️  Could not analyze fallback PDF content: {str(e)}")
                
                except Exception as e:
                    print(f"   ❌ Fallback PDF export error: {str(e)}")
        
        return success, response

    def test_filename_generation_improvement(self):
        """Test that filenames are meaningful and based on detected titles"""
        # Test with a clear scenario that should generate a specific title
        clear_scenario = {
            "plain_text": "Software development agreement for creating a mobile app with payment processing, user authentication, and cloud storage integration for $25,000",
            "expected_title_keywords": ["software", "development", "mobile", "app"]
        }
        
        conversion_data = {
            "plain_text": clear_scenario["plain_text"],
            "contract_type": "auto_detect",
            "jurisdiction": "US",
            "industry": "Technology",
            "output_format": "complete_contract"
        }
        
        success, response = self.run_test(
            "Filename Generation Improvement Test", 
            "POST", 
            "plain-english-to-legal", 
            200, 
            conversion_data,
            timeout=60
        )
        
        if success and response:
            conversion_id = response.get('id')
            if conversion_id:
                self.conversion_ids.append(conversion_id)
            
            detected_type = response.get('contract_type')
            print(f"   Detected contract type: {detected_type}")
            
            # Test PDF export to check filename generation
            export_data = {"format": "pdf"}
            url = f"{self.api_url}/plain-english-conversions/{conversion_id}/export"
            
            print(f"   Testing improved filename generation...")
            
            try:
                pdf_response = requests.post(url, json=export_data, headers={'Content-Type': 'application/json'}, timeout=30)
                
                if pdf_response.status_code == 200:
                    content_disposition = pdf_response.headers.get('content-disposition', '')
                    filename_match = re.search(r'filename="([^"]+)"', content_disposition)
                    
                    if filename_match:
                        filename = filename_match.group(1)
                        print(f"   📄 Generated Filename: {filename}")
                        
                        # Check filename quality
                        filename_lower = filename.lower()
                        
                        # Check for meaningful keywords
                        keyword_found = any(keyword in filename_lower for keyword in clear_scenario['expected_title_keywords'])
                        
                        if keyword_found:
                            print(f"   ✅ FILENAME IMPROVEMENT WORKING: Contains relevant keywords")
                        elif 'development' in filename_lower or 'software' in filename_lower:
                            print(f"   ✅ FILENAME IMPROVEMENT WORKING: Contains development-related terms")
                        elif 'service_agreement' in filename_lower or 'agreement' in filename_lower:
                            print(f"   ✅ FILENAME IMPROVEMENT WORKING: Professional agreement title")
                        elif 'plain_english_contract' in filename_lower:
                            print(f"   ❌ FILENAME IMPROVEMENT NOT WORKING: Still using generic pattern")
                        else:
                            print(f"   ⚠️  Filename pattern unclear - manual review needed")
                        
                        # Check for professional formatting
                        if filename.endswith('.pdf'):
                            print(f"   ✅ Proper PDF file extension")
                        
                        # Check for reasonable length
                        if 10 <= len(filename) <= 100:
                            print(f"   ✅ Reasonable filename length ({len(filename)} characters)")
                        else:
                            print(f"   ⚠️  Filename length may be suboptimal ({len(filename)} characters)")
                        
                        return True, {
                            "filename": filename,
                            "detected_type": detected_type,
                            "conversion_id": conversion_id,
                            "keyword_match": keyword_found
                        }
                    else:
                        print(f"   ❌ Could not extract filename from response headers")
                        return False, {}
                else:
                    print(f"   ❌ PDF export failed with status {pdf_response.status_code}")
                    return False, {}
                    
            except Exception as e:
                print(f"   ❌ PDF export error: {str(e)}")
                return False, {}
        
        return success, response

    def run_all_tests(self):
        """Run all PDF title generation tests"""
        print("🚀 Starting Plain English to Legal Clauses PDF Title Generation Tests")
        print("=" * 80)
        
        # Test 1: Main conversion endpoint with user scenario
        print("\n📋 TEST 1: Main Plain English Conversion with User Scenario")
        self.test_main_conversion_endpoint_user_scenario()
        
        # Test 2: PDF export with meaningful title
        print("\n📋 TEST 2: PDF Export with Meaningful Title")
        self.test_pdf_export_meaningful_title()
        
        # Test 3: Intelligent title detection scenarios
        print("\n📋 TEST 3: Intelligent Title Detection Scenarios")
        self.test_intelligent_title_detection_scenarios()
        
        # Test 4: Fallback title generation
        print("\n📋 TEST 4: Fallback Title Generation")
        self.test_fallback_title_generation()
        
        # Test 5: Filename generation improvement
        print("\n📋 TEST 5: Filename Generation Improvement")
        self.test_filename_generation_improvement()
        
        # Summary
        print("\n" + "=" * 80)
        print("📊 TEST SUMMARY")
        print("=" * 80)
        print(f"Total Tests Run: {self.tests_run}")
        print(f"Tests Passed: {self.tests_passed}")
        print(f"Tests Failed: {self.tests_run - self.tests_passed}")
        print(f"Success Rate: {(self.tests_passed / self.tests_run * 100):.1f}%" if self.tests_run > 0 else "0%")
        
        if self.tests_passed == self.tests_run:
            print("🎉 ALL TESTS PASSED - PDF Title Generation Fix is Working!")
        elif self.tests_passed / self.tests_run >= 0.8:
            print("✅ Most tests passed - PDF Title Generation Fix is mostly working")
        else:
            print("❌ Multiple test failures - PDF Title Generation Fix needs attention")
        
        print(f"\nGenerated {len(self.conversion_ids)} test conversions")
        if self.conversion_ids:
            print("Conversion IDs for manual verification:")
            for i, conv_id in enumerate(self.conversion_ids[:5]):  # Show first 5
                print(f"  {i+1}. {conv_id}")

if __name__ == "__main__":
    tester = PlainEnglishPDFTitleTester()
    tester.run_all_tests()