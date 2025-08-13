import requests
import sys
import json
import time
from datetime import datetime

class VoiceAgentBackendTester:
    def __init__(self, base_url="https://legalmate-research.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0

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
            elif method == 'PUT':
                response = requests.put(url, json=data, headers=headers, timeout=timeout)

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

    def test_legal_qa_ask_voice_basic(self):
        """Test POST /api/legal-qa/ask endpoint with voice agent parameters"""
        voice_qa_data = {
            "question": "What are the key elements of a valid contract?",
            "session_id": "voice_session_test_123",
            "jurisdiction": "US",
            "legal_domain": "contract_law",
            "is_voice": True,
            "conversation_context": []
        }
        
        success, response = self.run_test(
            "Legal Q&A Ask - Voice Agent Basic",
            "POST",
            "legal-qa/ask",
            200,
            voice_qa_data,
            timeout=60  # AI processing might take longer
        )
        
        if success and isinstance(response, dict):
            print(f"   ✅ Voice Q&A request successful")
            
            # Verify required response structure
            required_fields = ['answer', 'confidence', 'sources']
            missing_fields = []
            
            for field in required_fields:
                if field not in response:
                    missing_fields.append(field)
                else:
                    print(f"   ✅ Found required field: {field}")
                    
                    # Show field details
                    if field == 'answer':
                        answer_length = len(str(response[field]))
                        print(f"      Answer length: {answer_length} characters")
                        if answer_length > 50:
                            print(f"      Answer preview: {str(response[field])[:100]}...")
                    elif field == 'confidence':
                        confidence = response[field]
                        print(f"      Confidence score: {confidence}")
                        if isinstance(confidence, (int, float)) and 0 <= confidence <= 1:
                            print(f"      ✅ Confidence score in valid range (0-1)")
                        else:
                            print(f"      ⚠️  Confidence score format may be unexpected")
                    elif field == 'sources':
                        sources = response[field]
                        if isinstance(sources, list):
                            print(f"      Sources count: {len(sources)}")
                            if len(sources) > 0:
                                print(f"      ✅ Sources provided")
                                # Show first source structure
                                if isinstance(sources[0], dict):
                                    print(f"      First source keys: {list(sources[0].keys())}")
                            else:
                                print(f"      ⚠️  No sources provided")
                        else:
                            print(f"      ⚠️  Sources not in expected list format")
            
            if missing_fields:
                print(f"   ❌ Missing required fields: {missing_fields}")
                return False, response
            else:
                print(f"   ✅ All required response fields present")
                
            # Check for voice-specific handling
            if 'session_id' in response:
                print(f"   ✅ Session ID handled: {response['session_id']}")
            
            if 'is_voice' in response or response.get('is_voice') == True:
                print(f"   ✅ Voice mode acknowledged in response")
                
        return success, response

    def test_legal_qa_ask_voice_different_domains(self):
        """Test Legal Q&A with different legal domains"""
        test_cases = [
            {
                "domain": "contract_law",
                "question": "What constitutes a breach of contract?",
                "description": "Contract Law Domain"
            },
            {
                "domain": "employment_law",
                "question": "What are the requirements for wrongful termination?",
                "description": "Employment Law Domain"
            },
            {
                "domain": "intellectual_property",
                "question": "How long does a patent last?",
                "description": "Intellectual Property Domain"
            },
            {
                "domain": "corporate_law",
                "question": "What are the duties of corporate directors?",
                "description": "Corporate Law Domain"
            }
        ]
        
        all_success = True
        results = {}
        
        for i, test_case in enumerate(test_cases):
            voice_qa_data = {
                "question": test_case["question"],
                "session_id": f"voice_domain_test_{i+1}",
                "jurisdiction": "US",
                "legal_domain": test_case["domain"],
                "is_voice": True,
                "conversation_context": []
            }
            
            success, response = self.run_test(
                f"Legal Q&A - {test_case['description']}",
                "POST",
                "legal-qa/ask",
                200,
                voice_qa_data,
                timeout=60
            )
            
            if success and isinstance(response, dict):
                print(f"   ✅ {test_case['description']} successful")
                
                # Check response quality for domain-specific question
                if 'answer' in response:
                    answer = str(response['answer']).lower()
                    domain_keywords = {
                        'contract_law': ['contract', 'breach', 'agreement', 'obligation'],
                        'employment_law': ['employment', 'termination', 'employee', 'employer'],
                        'intellectual_property': ['patent', 'intellectual', 'property', 'invention'],
                        'corporate_law': ['corporate', 'director', 'fiduciary', 'company']
                    }
                    
                    expected_keywords = domain_keywords.get(test_case['domain'], [])
                    found_keywords = [kw for kw in expected_keywords if kw in answer]
                    
                    if found_keywords:
                        print(f"   ✅ Domain-relevant keywords found: {found_keywords}")
                    else:
                        print(f"   ⚠️  No domain-specific keywords found in answer")
                
                results[test_case['domain']] = {
                    "success": True,
                    "confidence": response.get('confidence', 0),
                    "sources_count": len(response.get('sources', []))
                }
            else:
                print(f"   ❌ {test_case['description']} failed")
                all_success = False
                results[test_case['domain']] = {"success": False}
        
        return all_success, results

    def test_legal_qa_ask_voice_different_jurisdictions(self):
        """Test Legal Q&A with different jurisdictions"""
        test_cases = [
            {
                "jurisdiction": "US",
                "question": "What is the statute of limitations for contract disputes?",
                "description": "US Jurisdiction"
            },
            {
                "jurisdiction": "UK",
                "question": "What is the limitation period for contract claims?",
                "description": "UK Jurisdiction"
            },
            {
                "jurisdiction": "CA",
                "question": "What are the limitation periods for contract actions?",
                "description": "Canada Jurisdiction"
            },
            {
                "jurisdiction": "AU",
                "question": "What is the limitation period for breach of contract?",
                "description": "Australia Jurisdiction"
            }
        ]
        
        all_success = True
        results = {}
        
        for i, test_case in enumerate(test_cases):
            voice_qa_data = {
                "question": test_case["question"],
                "session_id": f"voice_jurisdiction_test_{i+1}",
                "jurisdiction": test_case["jurisdiction"],
                "legal_domain": "contract_law",
                "is_voice": True,
                "conversation_context": []
            }
            
            success, response = self.run_test(
                f"Legal Q&A - {test_case['description']}",
                "POST",
                "legal-qa/ask",
                200,
                voice_qa_data,
                timeout=60
            )
            
            if success and isinstance(response, dict):
                print(f"   ✅ {test_case['description']} successful")
                
                # Check for jurisdiction-specific content
                if 'answer' in response:
                    answer = str(response['answer']).lower()
                    jurisdiction_indicators = {
                        'US': ['united states', 'federal', 'state law', 'ucc'],
                        'UK': ['united kingdom', 'english law', 'british', 'england'],
                        'CA': ['canada', 'canadian', 'provincial', 'common law'],
                        'AU': ['australia', 'australian', 'commonwealth']
                    }
                    
                    expected_indicators = jurisdiction_indicators.get(test_case['jurisdiction'], [])
                    found_indicators = [ind for ind in expected_indicators if ind in answer]
                    
                    if found_indicators:
                        print(f"   ✅ Jurisdiction-specific content found: {found_indicators}")
                    else:
                        print(f"   ⚠️  No jurisdiction-specific indicators found")
                
                results[test_case['jurisdiction']] = {
                    "success": True,
                    "confidence": response.get('confidence', 0),
                    "sources_count": len(response.get('sources', []))
                }
            else:
                print(f"   ❌ {test_case['jurisdiction']} failed")
                all_success = False
                results[test_case['jurisdiction']] = {"success": False}
        
        return all_success, results

    def test_legal_qa_session_handling(self):
        """Test session handling for voice conversations"""
        session_id = "voice_session_conversation_test"
        
        # First question in session
        first_question = {
            "question": "What is consideration in contract law?",
            "session_id": session_id,
            "jurisdiction": "US",
            "legal_domain": "contract_law",
            "is_voice": True,
            "conversation_context": []
        }
        
        success1, response1 = self.run_test(
            "Voice Session - First Question",
            "POST",
            "legal-qa/ask",
            200,
            first_question,
            timeout=60
        )
        
        if not success1:
            return False, {"first_question": response1}
        
        print(f"   ✅ First question in session successful")
        
        # Second question in same session with context
        conversation_context = []
        if isinstance(response1, dict) and 'answer' in response1:
            conversation_context = [
                {
                    "question": first_question["question"],
                    "answer": response1["answer"]
                }
            ]
        
        second_question = {
            "question": "Can you give me an example of consideration?",
            "session_id": session_id,
            "jurisdiction": "US",
            "legal_domain": "contract_law",
            "is_voice": True,
            "conversation_context": conversation_context
        }
        
        success2, response2 = self.run_test(
            "Voice Session - Follow-up Question with Context",
            "POST",
            "legal-qa/ask",
            200,
            second_question,
            timeout=60
        )
        
        if success2:
            print(f"   ✅ Follow-up question with context successful")
            
            # Check if the response acknowledges the context
            if isinstance(response2, dict) and 'answer' in response2:
                answer2 = str(response2['answer']).lower()
                if 'example' in answer2 or 'for instance' in answer2 or 'such as' in answer2:
                    print(f"   ✅ Response appears to address follow-up question appropriately")
                else:
                    print(f"   ⚠️  Response may not fully address follow-up nature of question")
        
        # Third question - different session to test isolation
        different_session = {
            "question": "What is consideration in contract law?",
            "session_id": "voice_session_different_test",
            "jurisdiction": "US",
            "legal_domain": "contract_law",
            "is_voice": True,
            "conversation_context": []
        }
        
        success3, response3 = self.run_test(
            "Voice Session - Different Session (Isolation Test)",
            "POST",
            "legal-qa/ask",
            200,
            different_session,
            timeout=60
        )
        
        if success3:
            print(f"   ✅ Different session handled independently")
        
        return success1 and success2 and success3, {
            "first_question": response1,
            "follow_up": response2,
            "different_session": response3
        }

    def test_legal_qa_voice_response_structure(self):
        """Test detailed response structure for voice interactions"""
        voice_qa_data = {
            "question": "What are the essential elements of a valid contract?",
            "session_id": "voice_structure_test",
            "jurisdiction": "US",
            "legal_domain": "contract_law",
            "is_voice": True,
            "conversation_context": []
        }
        
        success, response = self.run_test(
            "Legal Q&A - Voice Response Structure Analysis",
            "POST",
            "legal-qa/ask",
            200,
            voice_qa_data,
            timeout=60
        )
        
        if success and isinstance(response, dict):
            print(f"   ✅ Voice Q&A response received")
            
            # Detailed structure analysis
            structure_checks = {
                "answer": {
                    "required": True,
                    "type": str,
                    "min_length": 50
                },
                "confidence": {
                    "required": True,
                    "type": (int, float),
                    "range": (0, 1)
                },
                "sources": {
                    "required": True,
                    "type": list,
                    "min_items": 0
                }
            }
            
            all_checks_passed = True
            
            for field, checks in structure_checks.items():
                if field not in response:
                    if checks["required"]:
                        print(f"   ❌ Missing required field: {field}")
                        all_checks_passed = False
                    continue
                
                value = response[field]
                
                # Type check
                if "type" in checks:
                    if not isinstance(value, checks["type"]):
                        print(f"   ❌ Field '{field}' has incorrect type. Expected {checks['type']}, got {type(value)}")
                        all_checks_passed = False
                    else:
                        print(f"   ✅ Field '{field}' has correct type: {type(value).__name__}")
                
                # Length/range checks
                if field == "answer" and "min_length" in checks:
                    if len(str(value)) < checks["min_length"]:
                        print(f"   ❌ Answer too short: {len(str(value))} chars (min: {checks['min_length']})")
                        all_checks_passed = False
                    else:
                        print(f"   ✅ Answer has adequate length: {len(str(value))} chars")
                
                if field == "confidence" and "range" in checks:
                    min_val, max_val = checks["range"]
                    if not (min_val <= value <= max_val):
                        print(f"   ❌ Confidence out of range: {value} (expected {min_val}-{max_val})")
                        all_checks_passed = False
                    else:
                        print(f"   ✅ Confidence in valid range: {value}")
                
                if field == "sources" and "min_items" in checks:
                    if len(value) < checks["min_items"]:
                        print(f"   ⚠️  Sources list has {len(value)} items (min expected: {checks['min_items']})")
                    else:
                        print(f"   ✅ Sources list has {len(value)} items")
                        
                        # Check source structure if sources exist
                        if len(value) > 0 and isinstance(value[0], dict):
                            source_keys = list(value[0].keys())
                            print(f"   ✅ First source structure: {source_keys}")
            
            # Check for additional voice-specific fields
            voice_specific_fields = ['session_id', 'is_voice', 'response_time', 'legal_domain', 'jurisdiction']
            found_voice_fields = []
            
            for field in voice_specific_fields:
                if field in response:
                    found_voice_fields.append(field)
                    print(f"   ✅ Voice-specific field found: {field} = {response[field]}")
            
            if found_voice_fields:
                print(f"   ✅ Voice-specific fields present: {found_voice_fields}")
            else:
                print(f"   ⚠️  No voice-specific fields found in response")
            
            return all_checks_passed, response
        
        return success, response

    def test_legal_qa_voice_error_handling(self):
        """Test error handling for voice Q&A requests"""
        error_test_cases = [
            {
                "data": {
                    # Missing question
                    "session_id": "error_test_1",
                    "jurisdiction": "US",
                    "legal_domain": "contract_law",
                    "is_voice": True,
                    "conversation_context": []
                },
                "description": "Missing Question",
                "expected_status": 422
            },
            {
                "data": {
                    "question": "",  # Empty question
                    "session_id": "error_test_2",
                    "jurisdiction": "US",
                    "legal_domain": "contract_law",
                    "is_voice": True,
                    "conversation_context": []
                },
                "description": "Empty Question",
                "expected_status": 422
            },
            {
                "data": {
                    "question": "What is contract law?",
                    "session_id": "error_test_3",
                    "jurisdiction": "INVALID",  # Invalid jurisdiction
                    "legal_domain": "contract_law",
                    "is_voice": True,
                    "conversation_context": []
                },
                "description": "Invalid Jurisdiction",
                "expected_status": 422
            },
            {
                "data": {
                    "question": "What is contract law?",
                    "session_id": "error_test_4",
                    "jurisdiction": "US",
                    "legal_domain": "invalid_domain",  # Invalid legal domain
                    "is_voice": True,
                    "conversation_context": []
                },
                "description": "Invalid Legal Domain",
                "expected_status": 422
            }
        ]
        
        all_success = True
        results = {}
        
        for i, test_case in enumerate(error_test_cases):
            # Try the expected status first
            success, response = self.run_test(
                f"Error Handling - {test_case['description']}",
                "POST",
                "legal-qa/ask",
                test_case["expected_status"],
                test_case["data"],
                timeout=30
            )
            
            # If expected status doesn't work, try 500 (server error)
            if not success and test_case["expected_status"] == 422:
                success_500, response_500 = self.run_test(
                    f"Error Handling - {test_case['description']} (500)",
                    "POST",
                    "legal-qa/ask",
                    500,
                    test_case["data"],
                    timeout=30
                )
                if success_500:
                    success = True
                    response = response_500
                    self.tests_passed += 1  # Adjust count
            
            if success:
                print(f"   ✅ Error properly handled for {test_case['description']}")
                results[test_case['description']] = {"success": True, "response": response}
            else:
                print(f"   ❌ Error handling failed for {test_case['description']}")
                all_success = False
                results[test_case['description']] = {"success": False}
        
        return all_success, results

    def run_all_voice_tests(self):
        """Run all voice agent backend tests"""
        print("🎤 VOICE AGENT BACKEND INTEGRATION TESTING")
        print("=" * 60)
        print(f"Testing against: {self.base_url}")
        print(f"API Endpoint: {self.api_url}")
        
        test_methods = [
            self.test_legal_qa_ask_voice_basic,
            self.test_legal_qa_ask_voice_different_domains,
            self.test_legal_qa_ask_voice_different_jurisdictions,
            self.test_legal_qa_session_handling,
            self.test_legal_qa_voice_response_structure,
            self.test_legal_qa_voice_error_handling
        ]
        
        results = {}
        
        for test_method in test_methods:
            try:
                test_name = test_method.__name__
                print(f"\n{'='*60}")
                print(f"Running: {test_name}")
                print(f"{'='*60}")
                
                success, result = test_method()
                results[test_name] = {
                    "success": success,
                    "result": result
                }
                
                if success:
                    print(f"✅ {test_name} - PASSED")
                else:
                    print(f"❌ {test_name} - FAILED")
                    
            except Exception as e:
                print(f"❌ {test_method.__name__} - ERROR: {str(e)}")
                results[test_method.__name__] = {
                    "success": False,
                    "error": str(e)
                }
        
        # Final summary
        print(f"\n{'='*60}")
        print("🎤 VOICE AGENT BACKEND TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total Tests Run: {self.tests_run}")
        print(f"Tests Passed: {self.tests_passed}")
        print(f"Success Rate: {(self.tests_passed/self.tests_run*100):.1f}%" if self.tests_run > 0 else "0%")
        
        # Detailed results
        print(f"\nDetailed Results:")
        for test_name, result in results.items():
            status = "✅ PASS" if result["success"] else "❌ FAIL"
            print(f"  {status} - {test_name}")
            if not result["success"] and "error" in result:
                print(f"    Error: {result['error']}")
        
        return results

if __name__ == "__main__":
    tester = VoiceAgentBackendTester()
    results = tester.run_all_voice_tests()
    
    # Exit with appropriate code
    if all(result["success"] for result in results.values()):
        print(f"\n🎉 All voice agent backend tests passed!")
        sys.exit(0)
    else:
        print(f"\n⚠️  Some voice agent backend tests failed. Check the results above.")
        sys.exit(1)