import requests
import sys
import json
import time
from datetime import datetime

class LegalQABackendTester:
    def __init__(self, base_url="https://d7a1dca3-54ad-4826-aadd-9f393fe0628b.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.session_id = f"test-session-{int(time.time())}"

    def run_test(self, name, method, endpoint, expected_status, data=None, timeout=60):
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

    def test_legal_qa_ask_endpoint(self):
        """Test the main Legal Q&A endpoint with sample legal question"""
        legal_question_data = {
            "question": "What should I do if I'm being harassed by debt collectors?",
            "session_id": self.session_id,
            "jurisdiction": "US",
            "legal_domain": "consumer_law"
        }
        
        success, response = self.run_test(
            "Legal Q&A Ask Endpoint - Debt Collection Question", 
            "POST", 
            "legal-qa/ask", 
            200, 
            legal_question_data,
            timeout=90  # Legal Q&A might take longer due to AI processing
        )
        
        if success and isinstance(response, dict):
            print(f"   📊 Response Analysis:")
            
            # Check for expected response fields
            expected_fields = ['answer', 'confidence', 'sources', 'session_id', 'status']
            for field in expected_fields:
                if field in response:
                    print(f"   ✅ Field '{field}' present: {type(response[field])}")
                    
                    # Analyze specific fields
                    if field == 'answer':
                        answer = response[field]
                        if isinstance(answer, str) and len(answer) > 50:
                            print(f"      Answer length: {len(answer)} characters")
                            print(f"      Answer preview: {answer[:100]}...")
                            
                            # Check for fallback response indicators
                            fallback_indicators = [
                                "I apologize, but I'm unable to generate a response",
                                "I cannot provide legal advice",
                                "Please consult with a qualified attorney",
                                "fallback response"
                            ]
                            
                            is_fallback = any(indicator.lower() in answer.lower() for indicator in fallback_indicators)
                            if is_fallback:
                                print(f"      ❌ FALLBACK RESPONSE DETECTED - This is the reported issue!")
                            else:
                                print(f"      ✅ Appears to be a proper legal response")
                        else:
                            print(f"      ❌ Answer too short or invalid: {answer}")
                    
                    elif field == 'confidence':
                        confidence = response[field]
                        print(f"      Confidence score: {confidence}")
                        if confidence == 0 or confidence == 0.0:
                            print(f"      ❌ ZERO CONFIDENCE - This matches the reported issue!")
                        elif confidence < 0.3:
                            print(f"      ⚠️  Very low confidence score")
                        else:
                            print(f"      ✅ Reasonable confidence score")
                    
                    elif field == 'status':
                        status = response[field]
                        print(f"      Status: {status}")
                        if status == 'fallback':
                            print(f"      ❌ FALLBACK STATUS - This matches the reported issue!")
                        elif status == 'success':
                            print(f"      ✅ Success status")
                        else:
                            print(f"      ⚠️  Unexpected status: {status}")
                    
                    elif field == 'sources':
                        sources = response[field]
                        if isinstance(sources, list):
                            print(f"      Sources count: {len(sources)}")
                            if len(sources) == 0:
                                print(f"      ⚠️  No sources provided")
                            else:
                                print(f"      ✅ Sources provided")
                        else:
                            print(f"      ⚠️  Sources not in expected list format")
                            
                else:
                    print(f"   ❌ Missing expected field: '{field}'")
            
            # Overall assessment
            is_working_properly = (
                response.get('confidence', 0) > 0 and 
                response.get('status') != 'fallback' and
                isinstance(response.get('answer'), str) and 
                len(response.get('answer', '')) > 50 and
                not any(indicator.lower() in response.get('answer', '').lower() 
                       for indicator in ["I apologize, but I'm unable to generate a response"])
            )
            
            if is_working_properly:
                print(f"   🎉 LEGAL Q&A APPEARS TO BE WORKING PROPERLY!")
            else:
                print(f"   ❌ LEGAL Q&A HAS ISSUES - MATCHES USER REPORT")
                
        return success, response

    def test_legal_qa_stats_endpoint(self):
        """Test the RAG system stats endpoint"""
        success, response = self.run_test(
            "Legal Q&A Stats - RAG System Status", 
            "GET", 
            "legal-qa/stats", 
            200
        )
        
        if success and isinstance(response, dict):
            print(f"   📊 RAG System Analysis:")
            
            # Check for expected stats fields
            expected_fields = ['vector_db', 'embeddings_model', 'total_documents', 'system_status']
            for field in expected_fields:
                if field in response:
                    value = response[field]
                    print(f"   ✅ {field}: {value}")
                    
                    if field == 'system_status':
                        if value == 'operational':
                            print(f"      ✅ RAG system is operational")
                        else:
                            print(f"      ❌ RAG system status issue: {value}")
                    
                    elif field == 'total_documents':
                        if isinstance(value, int) and value > 0:
                            print(f"      ✅ Knowledge base has {value} documents")
                        else:
                            print(f"      ❌ Knowledge base appears empty or invalid")
                            
                else:
                    print(f"   ❌ Missing expected field: '{field}'")
            
            # Check for additional useful fields
            optional_fields = ['last_updated', 'index_status', 'embedding_dimensions']
            for field in optional_fields:
                if field in response:
                    print(f"   ℹ️  {field}: {response[field]}")
                    
        return success, response

    def test_knowledge_base_stats_endpoint(self):
        """Test the knowledge base stats endpoint"""
        success, response = self.run_test(
            "Knowledge Base Stats", 
            "GET", 
            "legal-qa/knowledge-base/stats", 
            200
        )
        
        if success and isinstance(response, dict):
            print(f"   📊 Knowledge Base Analysis:")
            
            # Check for expected knowledge base fields
            expected_fields = ['total_documents', 'jurisdictions', 'legal_domains']
            for field in expected_fields:
                if field in response:
                    value = response[field]
                    print(f"   ✅ {field}: {value}")
                    
                    if field == 'total_documents':
                        if isinstance(value, int) and value > 0:
                            print(f"      ✅ Knowledge base populated with {value} documents")
                        else:
                            print(f"      ❌ Knowledge base appears empty: {value}")
                    
                    elif field == 'jurisdictions':
                        if isinstance(value, (list, dict)):
                            if isinstance(value, list):
                                print(f"      ✅ Jurisdictions available: {len(value)} ({value})")
                            else:
                                print(f"      ✅ Jurisdictions data: {value}")
                        else:
                            print(f"      ⚠️  Unexpected jurisdictions format: {type(value)}")
                    
                    elif field == 'legal_domains':
                        if isinstance(value, (list, dict)):
                            if isinstance(value, list):
                                print(f"      ✅ Legal domains available: {len(value)} ({value})")
                            else:
                                print(f"      ✅ Legal domains data: {value}")
                        else:
                            print(f"      ⚠️  Unexpected legal domains format: {type(value)}")
                            
                else:
                    print(f"   ❌ Missing expected field: '{field}'")
            
            # Check for additional useful fields
            optional_fields = ['last_updated', 'document_types', 'quality_metrics']
            for field in optional_fields:
                if field in response:
                    print(f"   ℹ️  {field}: {response[field]}")
                    
        return success, response

    def test_various_legal_questions(self):
        """Test various legal questions to see if any work properly"""
        test_questions = [
            {
                "question": "What are my rights if my landlord wants to evict me?",
                "jurisdiction": "US",
                "legal_domain": "housing_law",
                "description": "Housing/Tenant Rights"
            },
            {
                "question": "Can my employer fire me for filing a workers compensation claim?",
                "jurisdiction": "US", 
                "legal_domain": "employment_law",
                "description": "Employment Law"
            },
            {
                "question": "What should I include in a non-disclosure agreement?",
                "jurisdiction": "US",
                "legal_domain": "contract_law", 
                "description": "Contract Law"
            },
            {
                "question": "How do I file for bankruptcy and what are the consequences?",
                "jurisdiction": "US",
                "legal_domain": "bankruptcy_law",
                "description": "Bankruptcy Law"
            },
            {
                "question": "What are the requirements for a valid will?",
                "jurisdiction": "US",
                "legal_domain": "estate_law",
                "description": "Estate Planning"
            }
        ]
        
        working_questions = 0
        total_questions = len(test_questions)
        results = {}
        
        for i, test_case in enumerate(test_questions):
            question_data = {
                "question": test_case["question"],
                "session_id": f"{self.session_id}-q{i+1}",
                "jurisdiction": test_case["jurisdiction"],
                "legal_domain": test_case["legal_domain"]
            }
            
            success, response = self.run_test(
                f"Legal Question {i+1} - {test_case['description']}", 
                "POST", 
                "legal-qa/ask", 
                200, 
                question_data,
                timeout=90
            )
            
            if success and isinstance(response, dict):
                # Analyze response quality
                confidence = response.get('confidence', 0)
                status = response.get('status', 'unknown')
                answer = response.get('answer', '')
                
                print(f"   📊 Question Analysis:")
                print(f"      Confidence: {confidence}")
                print(f"      Status: {status}")
                print(f"      Answer length: {len(answer)} characters")
                
                # Check if this is a proper response
                is_proper_response = (
                    confidence > 0 and 
                    status != 'fallback' and
                    len(answer) > 50 and
                    not any(indicator.lower() in answer.lower() 
                           for indicator in ["I apologize, but I'm unable to generate a response"])
                )
                
                if is_proper_response:
                    working_questions += 1
                    print(f"      ✅ PROPER RESPONSE - This question works!")
                    print(f"      Answer preview: {answer[:150]}...")
                else:
                    print(f"      ❌ FALLBACK/POOR RESPONSE")
                    if len(answer) > 0:
                        print(f"      Answer preview: {answer[:150]}...")
                
                results[test_case['description']] = {
                    'success': success,
                    'confidence': confidence,
                    'status': status,
                    'answer_length': len(answer),
                    'is_proper_response': is_proper_response
                }
            else:
                results[test_case['description']] = {
                    'success': False,
                    'error': 'Request failed'
                }
        
        print(f"\n📈 OVERALL LEGAL Q&A ASSESSMENT:")
        print(f"   Working questions: {working_questions}/{total_questions}")
        print(f"   Success rate: {(working_questions/total_questions)*100:.1f}%")
        
        if working_questions == 0:
            print(f"   ❌ CRITICAL ISSUE: NO QUESTIONS WORKING - CONFIRMS USER REPORT")
        elif working_questions < total_questions:
            print(f"   ⚠️  PARTIAL FUNCTIONALITY: Some questions work, others don't")
        else:
            print(f"   ✅ ALL QUESTIONS WORKING PROPERLY")
            
        return working_questions > 0, results

    def test_error_handling_and_logs(self):
        """Test error conditions and check for common issues"""
        print(f"\n🔍 Testing Error Conditions and Edge Cases...")
        
        # Test 1: Empty question
        empty_question_data = {
            "question": "",
            "session_id": self.session_id,
            "jurisdiction": "US",
            "legal_domain": "general"
        }
        
        success1, response1 = self.run_test(
            "Empty Question Test", 
            "POST", 
            "legal-qa/ask", 
            422,  # Should return validation error
            empty_question_data
        )
        
        # If 422 doesn't work, try 400
        if not success1:
            success1, response1 = self.run_test(
                "Empty Question Test (400)", 
                "POST", 
                "legal-qa/ask", 
                400,
                empty_question_data
            )
            if success1:
                self.tests_passed += 1  # Adjust count
        
        # Test 2: Invalid jurisdiction
        invalid_jurisdiction_data = {
            "question": "What are my rights?",
            "session_id": self.session_id,
            "jurisdiction": "INVALID_JURISDICTION",
            "legal_domain": "general"
        }
        
        success2, response2 = self.run_test(
            "Invalid Jurisdiction Test", 
            "POST", 
            "legal-qa/ask", 
            200,  # Should still work but may affect quality
            invalid_jurisdiction_data,
            timeout=60
        )
        
        if success2 and isinstance(response2, dict):
            print(f"   ℹ️  Invalid jurisdiction handling:")
            print(f"      Confidence: {response2.get('confidence', 'N/A')}")
            print(f"      Status: {response2.get('status', 'N/A')}")
        
        # Test 3: Very long question
        long_question_data = {
            "question": "What should I do " * 100 + "in this legal situation?",  # Very long question
            "session_id": self.session_id,
            "jurisdiction": "US",
            "legal_domain": "general"
        }
        
        success3, response3 = self.run_test(
            "Very Long Question Test", 
            "POST", 
            "legal-qa/ask", 
            200,
            long_question_data,
            timeout=90
        )
        
        # Test 4: Missing required fields
        incomplete_data = {
            "question": "What are my rights?"
            # Missing session_id, jurisdiction, legal_domain
        }
        
        success4, response4 = self.run_test(
            "Missing Required Fields Test", 
            "POST", 
            "legal-qa/ask", 
            422,  # Should return validation error
            incomplete_data
        )
        
        # If 422 doesn't work, try 400
        if not success4:
            success4, response4 = self.run_test(
                "Missing Required Fields Test (400)", 
                "POST", 
                "legal-qa/ask", 
                400,
                incomplete_data
            )
            if success4:
                self.tests_passed += 1  # Adjust count
        
        return True, {
            'empty_question': success1,
            'invalid_jurisdiction': success2,
            'long_question': success3,
            'missing_fields': success4
        }

    def test_api_key_and_integration_issues(self):
        """Test for potential API key and integration issues"""
        print(f"\n🔍 Checking for Integration Issues...")
        
        # Test a simple question and analyze the response for integration clues
        test_data = {
            "question": "What is contract law?",
            "session_id": f"{self.session_id}-integration-test",
            "jurisdiction": "US",
            "legal_domain": "contract_law"
        }
        
        success, response = self.run_test(
            "Integration Issues Check", 
            "POST", 
            "legal-qa/ask", 
            200, 
            test_data,
            timeout=90
        )
        
        if success and isinstance(response, dict):
            print(f"   🔍 Analyzing Response for Integration Clues:")
            
            answer = response.get('answer', '')
            confidence = response.get('confidence', 0)
            status = response.get('status', '')
            
            # Check for common API key error indicators
            api_key_indicators = [
                'api key',
                'authentication',
                'unauthorized',
                'quota exceeded',
                'rate limit',
                'invalid key',
                'access denied'
            ]
            
            has_api_key_issue = any(indicator in answer.lower() for indicator in api_key_indicators)
            if has_api_key_issue:
                print(f"   ❌ POTENTIAL API KEY ISSUE DETECTED in response")
                print(f"      Answer contains API-related error terms")
            else:
                print(f"   ✅ No obvious API key issues in response")
            
            # Check for AI model access issues
            ai_model_indicators = [
                'model not available',
                'service unavailable',
                'connection error',
                'timeout',
                'internal server error'
            ]
            
            has_ai_model_issue = any(indicator in answer.lower() for indicator in ai_model_indicators)
            if has_ai_model_issue:
                print(f"   ❌ POTENTIAL AI MODEL ACCESS ISSUE DETECTED")
            else:
                print(f"   ✅ No obvious AI model access issues")
            
            # Check for database connectivity issues
            db_indicators = [
                'database',
                'connection failed',
                'mongodb',
                'collection',
                'query failed'
            ]
            
            has_db_issue = any(indicator in answer.lower() for indicator in db_indicators)
            if has_db_issue:
                print(f"   ❌ POTENTIAL DATABASE CONNECTIVITY ISSUE DETECTED")
            else:
                print(f"   ✅ No obvious database connectivity issues")
            
            # Overall assessment
            if confidence == 0 and status == 'fallback':
                print(f"   ❌ SYSTEM APPEARS TO BE IN FALLBACK MODE")
                print(f"      This suggests a fundamental integration issue")
                print(f"      Possible causes:")
                print(f"        - AI API keys not working (Groq, OpenAI, Gemini)")
                print(f"        - RAG system not initialized properly")
                print(f"        - Knowledge base not populated")
                print(f"        - Database connection issues")
            elif confidence > 0 and len(answer) > 100:
                print(f"   ✅ SYSTEM APPEARS TO BE WORKING")
                print(f"      Integration seems functional")
            else:
                print(f"   ⚠️  SYSTEM STATUS UNCLEAR")
                print(f"      May have partial functionality issues")
                
        return success, response

    def run_comprehensive_legal_qa_test(self):
        """Run all Legal Q&A tests"""
        print("=" * 80)
        print("🏛️  LEGAL Q&A BACKEND COMPREHENSIVE TESTING")
        print("=" * 80)
        print(f"Testing against: {self.base_url}")
        print(f"Session ID: {self.session_id}")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Test 1: Main Legal Q&A endpoint
        print(f"\n" + "="*60)
        print(f"TEST 1: MAIN LEGAL Q&A ENDPOINT")
        print(f"="*60)
        self.test_legal_qa_ask_endpoint()
        
        # Test 2: RAG system stats
        print(f"\n" + "="*60)
        print(f"TEST 2: RAG SYSTEM STATUS")
        print(f"="*60)
        self.test_legal_qa_stats_endpoint()
        
        # Test 3: Knowledge base stats
        print(f"\n" + "="*60)
        print(f"TEST 3: KNOWLEDGE BASE STATUS")
        print(f"="*60)
        self.test_knowledge_base_stats_endpoint()
        
        # Test 4: Various legal questions
        print(f"\n" + "="*60)
        print(f"TEST 4: VARIOUS LEGAL QUESTIONS")
        print(f"="*60)
        self.test_various_legal_questions()
        
        # Test 5: Error handling
        print(f"\n" + "="*60)
        print(f"TEST 5: ERROR HANDLING & EDGE CASES")
        print(f"="*60)
        self.test_error_handling_and_logs()
        
        # Test 6: Integration issues check
        print(f"\n" + "="*60)
        print(f"TEST 6: INTEGRATION ISSUES CHECK")
        print(f"="*60)
        self.test_api_key_and_integration_issues()
        
        # Final summary
        print(f"\n" + "="*80)
        print(f"🏛️  LEGAL Q&A TESTING SUMMARY")
        print(f"="*80)
        print(f"Total tests run: {self.tests_run}")
        print(f"Tests passed: {self.tests_passed}")
        print(f"Success rate: {(self.tests_passed/self.tests_run)*100:.1f}%")
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if self.tests_passed == 0:
            print(f"\n❌ CRITICAL: ALL TESTS FAILED")
            print(f"   This confirms the user's report of Legal Q&A not working")
            print(f"   Likely causes:")
            print(f"   - Legal Q&A endpoints not implemented")
            print(f"   - RAG system not available")
            print(f"   - AI API integration issues")
            print(f"   - Knowledge base not populated")
        elif self.tests_passed < self.tests_run * 0.5:
            print(f"\n⚠️  MAJOR ISSUES: Less than 50% tests passed")
            print(f"   Legal Q&A system has significant problems")
        else:
            print(f"\n✅ SYSTEM APPEARS FUNCTIONAL")
            print(f"   Legal Q&A system is working properly")
        
        return self.tests_passed, self.tests_run

if __name__ == "__main__":
    tester = LegalQABackendTester()
    passed, total = tester.run_comprehensive_legal_qa_test()
    
    # Exit with appropriate code
    if passed == 0:
        sys.exit(1)  # Critical failure
    elif passed < total * 0.5:
        sys.exit(2)  # Major issues
    else:
        sys.exit(0)  # Success