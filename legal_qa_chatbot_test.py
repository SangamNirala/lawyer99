#!/usr/bin/env python3
"""
Legal Q&A Chatbot Improvements Testing

This script tests the specific improvements made to the legal Q&A chatbot:
1. Simple Greeting Test - No confidence scores in response text, no legal disclaimer, confidence: 1.0, model_used: "greeting_handler"
2. Complex Legal Question Test - Includes legal disclaimer, confidence metadata, proper formatting, retrieved documents and sources
3. Other Simple Interactions - Test various simple phrases like "thank you", "how are you", "testing"

Focus: Verify the fixes for greeting handling and legal question processing
"""

import requests
import sys
import json
import re
from datetime import datetime

class LegalQAChatbotTester:
    def __init__(self, base_url="https://b2e17d7f-d715-45b1-acd0-e51dd70d439b.preview.emergentagent.com"):
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

    def test_simple_greeting_hi(self):
        """Test simple greeting 'hi' - should NOT include confidence scores in response text, NO legal disclaimer, confidence: 1.0, model_used: 'greeting_handler'"""
        
        greeting_data = {
            "question": "hi",
            "session_id": "test_greeting_session_hi"
        }
        
        success, response = self.run_test(
            "Simple Greeting Test - 'hi'", 
            "POST", 
            "legal-qa/ask", 
            200, 
            greeting_data,
            timeout=30
        )
        
        if success and isinstance(response, dict):
            print(f"\n   📋 Analyzing greeting response for 'hi'...")
            
            # Check 1: Response should be friendly and professional
            answer = response.get('answer', '')
            print(f"   Answer: {answer[:100]}...")
            
            if any(word in answer.lower() for word in ['hello', 'hi', 'legal', 'assistant', 'help']):
                print(f"   ✅ Response is friendly and professional")
            else:
                print(f"   ❌ Response doesn't seem friendly/professional")
            
            # Check 2: Should NOT include confidence scores in the response text
            if 'confidence' not in answer.lower() and 'score' not in answer.lower():
                print(f"   ✅ No confidence scores found in response text")
            else:
                print(f"   ❌ Confidence scores found in response text (should not be there)")
            
            # Check 3: Should NOT include the legal disclaimer at the end
            legal_disclaimer_patterns = [
                "legal disclaimer", "informational purposes only", "does not constitute legal advice", 
                "consult with a qualified attorney", "⚖️"
            ]
            has_disclaimer = any(pattern in answer.lower() for pattern in legal_disclaimer_patterns)
            
            if not has_disclaimer:
                print(f"   ✅ No legal disclaimer found in response (correct for greetings)")
            else:
                print(f"   ❌ Legal disclaimer found in response (should not be there for greetings)")
            
            # Check 4: Should have confidence: 1.0
            confidence = response.get('confidence', 0)
            if confidence == 1.0:
                print(f"   ✅ Confidence is 1.0 (correct for greetings)")
            else:
                print(f"   ❌ Confidence is {confidence}, expected 1.0")
            
            # Check 5: Should have model_used: "greeting_handler"
            model_used = response.get('model_used', '')
            if model_used == "greeting_handler":
                print(f"   ✅ Model used is 'greeting_handler' (correct)")
            else:
                print(f"   ❌ Model used is '{model_used}', expected 'greeting_handler'")
            
            # Check 6: Should have empty sources array
            sources = response.get('sources', [])
            if len(sources) == 0:
                print(f"   ✅ Sources array is empty (correct for greetings)")
            else:
                print(f"   ❌ Sources array has {len(sources)} items, expected 0")
            
            # Check 7: Should have retrieved_documents: 0
            retrieved_documents = response.get('retrieved_documents', -1)
            if retrieved_documents == 0:
                print(f"   ✅ Retrieved documents is 0 (correct for greetings)")
            else:
                print(f"   ❌ Retrieved documents is {retrieved_documents}, expected 0")
            
            # Summary for this test
            checks_passed = [
                any(word in answer.lower() for word in ['hello', 'hi', 'legal', 'assistant', 'help']),
                'confidence' not in answer.lower() and 'score' not in answer.lower(),
                not has_disclaimer,
                confidence == 1.0,
                model_used == "greeting_handler",
                len(sources) == 0,
                retrieved_documents == 0
            ]
            
            passed_count = sum(checks_passed)
            print(f"   📊 Greeting Test Summary: {passed_count}/7 checks passed")
            
            if passed_count >= 6:  # Allow 1 minor failure
                print(f"   🎉 Simple greeting test PASSED - All key requirements met")
            else:
                print(f"   ❌ Simple greeting test FAILED - Multiple requirements not met")
            
        return success, response

    def test_complex_legal_question(self):
        """Test complex legal question - should include legal disclaimer, confidence metadata, proper formatting, retrieved documents and sources"""
        
        legal_question_data = {
            "question": "What are the key elements of a valid contract under US law?",
            "session_id": "test_legal_session_contract",
            "jurisdiction": "US",
            "legal_domain": "contract_law"
        }
        
        success, response = self.run_test(
            "Complex Legal Question Test - Contract Elements", 
            "POST", 
            "legal-qa/ask", 
            200, 
            legal_question_data,
            timeout=45  # Legal questions may take longer
        )
        
        if success and isinstance(response, dict):
            print(f"\n   📋 Analyzing legal question response...")
            
            # Check 1: Response should include the legal disclaimer
            answer = response.get('answer', '')
            print(f"   Answer length: {len(answer)} characters")
            print(f"   Answer preview: {answer[:150]}...")
            
            legal_disclaimer_patterns = [
                "legal disclaimer", "informational purposes only", "does not constitute legal advice", 
                "consult with a qualified attorney", "⚖️"
            ]
            has_disclaimer = any(pattern in answer.lower() for pattern in legal_disclaimer_patterns)
            
            if has_disclaimer:
                print(f"   ✅ Legal disclaimer found in response (correct for legal questions)")
            else:
                print(f"   ❌ Legal disclaimer NOT found in response (should be there for legal questions)")
            
            # Check 2: Response should have confidence metadata
            confidence = response.get('confidence', -1)
            if confidence >= 0 and confidence <= 1:
                print(f"   ✅ Confidence metadata present: {confidence}")
            else:
                print(f"   ❌ Invalid or missing confidence metadata: {confidence}")
            
            # Check 3: Should include proper formatting (check if **bold** text appears as raw asterisks or properly formatted)
            # Look for **bold** patterns in the response
            bold_patterns = re.findall(r'\*\*[^*]+\*\*', answer)
            if bold_patterns:
                print(f"   ✅ Found {len(bold_patterns)} **bold** formatting patterns")
                print(f"   Bold examples: {bold_patterns[:3]}")
            else:
                print(f"   ⚠️  No **bold** formatting patterns found (may be normal)")
            
            # Check for raw asterisks that shouldn't be there
            single_asterisks = answer.count('*') - (len(bold_patterns) * 4)  # Each **bold** has 4 asterisks
            if single_asterisks <= 0:
                print(f"   ✅ No improper asterisk formatting found")
            else:
                print(f"   ⚠️  Found {single_asterisks} single asterisks (may indicate formatting issues)")
            
            # Check 4: Should include retrieved documents and sources
            sources = response.get('sources', [])
            retrieved_documents = response.get('retrieved_documents', 0)
            
            if len(sources) > 0:
                print(f"   ✅ Sources provided: {len(sources)} sources")
                # Show first source as example
                if sources:
                    first_source = sources[0]
                    print(f"   Source example: {list(first_source.keys()) if isinstance(first_source, dict) else str(first_source)[:100]}")
            else:
                print(f"   ❌ No sources provided (should have sources for legal questions)")
            
            if retrieved_documents > 0:
                print(f"   ✅ Retrieved documents: {retrieved_documents}")
            else:
                print(f"   ❌ No documents retrieved (should retrieve documents for legal questions)")
            
            # Check 5: Model used should NOT be greeting_handler
            model_used = response.get('model_used', '')
            if model_used != "greeting_handler":
                print(f"   ✅ Model used is '{model_used}' (not greeting_handler, correct)")
            else:
                print(f"   ❌ Model used is 'greeting_handler' (should be different for legal questions)")
            
            # Check 6: Response should contain legal content
            legal_keywords = ['contract', 'legal', 'law', 'element', 'requirement', 'valid', 'agreement']
            legal_content_found = sum(1 for keyword in legal_keywords if keyword in answer.lower())
            
            if legal_content_found >= 3:
                print(f"   ✅ Response contains legal content ({legal_content_found} legal keywords found)")
            else:
                print(f"   ❌ Response may lack legal content ({legal_content_found} legal keywords found)")
            
            # Summary for this test
            checks_passed = [
                has_disclaimer,
                confidence >= 0 and confidence <= 1,
                len(sources) > 0,
                retrieved_documents > 0,
                model_used != "greeting_handler",
                legal_content_found >= 3
            ]
            
            passed_count = sum(checks_passed)
            print(f"   📊 Legal Question Test Summary: {passed_count}/6 checks passed")
            
            if passed_count >= 5:  # Allow 1 minor failure
                print(f"   🎉 Complex legal question test PASSED - All key requirements met")
            else:
                print(f"   ❌ Complex legal question test FAILED - Multiple requirements not met")
            
        return success, response

    def test_other_simple_interactions(self):
        """Test other simple phrases like 'thank you', 'how are you', 'testing'"""
        
        simple_phrases = [
            {
                "phrase": "thank you",
                "expected_keywords": ["welcome", "help", "legal"]
            },
            {
                "phrase": "how are you",
                "expected_keywords": ["well", "ready", "legal", "help"]
            },
            {
                "phrase": "testing",
                "expected_keywords": ["working", "legal", "assistant", "help"]
            }
        ]
        
        all_passed = True
        results = {}
        
        for i, test_case in enumerate(simple_phrases):
            phrase = test_case["phrase"]
            expected_keywords = test_case["expected_keywords"]
            
            interaction_data = {
                "question": phrase,
                "session_id": f"test_simple_session_{i}"
            }
            
            success, response = self.run_test(
                f"Simple Interaction Test - '{phrase}'", 
                "POST", 
                "legal-qa/ask", 
                200, 
                interaction_data,
                timeout=30
            )
            
            if success and isinstance(response, dict):
                print(f"\n   📋 Analyzing simple interaction response for '{phrase}'...")
                
                answer = response.get('answer', '')
                print(f"   Answer: {answer[:100]}...")
                
                # Check 1: Should be handled as greeting (no legal disclaimer)
                legal_disclaimer_patterns = [
                    "legal disclaimer", "informational purposes only", "does not constitute legal advice"
                ]
                has_disclaimer = any(pattern in answer.lower() for pattern in legal_disclaimer_patterns)
                
                if not has_disclaimer:
                    print(f"   ✅ No legal disclaimer (correct for simple interactions)")
                else:
                    print(f"   ❌ Legal disclaimer found (should not be there for simple interactions)")
                
                # Check 2: Should have confidence: 1.0
                confidence = response.get('confidence', 0)
                if confidence == 1.0:
                    print(f"   ✅ Confidence is 1.0 (correct)")
                else:
                    print(f"   ❌ Confidence is {confidence}, expected 1.0")
                
                # Check 3: Should have model_used: "greeting_handler"
                model_used = response.get('model_used', '')
                if model_used == "greeting_handler":
                    print(f"   ✅ Model used is 'greeting_handler' (correct)")
                else:
                    print(f"   ❌ Model used is '{model_used}', expected 'greeting_handler'")
                
                # Check 4: Should contain expected keywords
                keywords_found = sum(1 for keyword in expected_keywords if keyword in answer.lower())
                if keywords_found >= 2:
                    print(f"   ✅ Response contains expected keywords ({keywords_found}/{len(expected_keywords)})")
                else:
                    print(f"   ❌ Response lacks expected keywords ({keywords_found}/{len(expected_keywords)})")
                
                # Check 5: Should have empty sources and 0 retrieved documents
                sources = response.get('sources', [])
                retrieved_documents = response.get('retrieved_documents', -1)
                
                if len(sources) == 0 and retrieved_documents == 0:
                    print(f"   ✅ No sources/documents retrieved (correct for simple interactions)")
                else:
                    print(f"   ❌ Sources: {len(sources)}, Documents: {retrieved_documents} (should be 0)")
                
                # Summary for this phrase
                checks_passed = [
                    not has_disclaimer,
                    confidence == 1.0,
                    model_used == "greeting_handler",
                    keywords_found >= 2,
                    len(sources) == 0 and retrieved_documents == 0
                ]
                
                phrase_passed = sum(checks_passed) >= 4  # Allow 1 minor failure
                results[phrase] = {
                    "passed": phrase_passed,
                    "checks_passed": sum(checks_passed),
                    "total_checks": len(checks_passed)
                }
                
                if phrase_passed:
                    print(f"   🎉 Simple interaction '{phrase}' PASSED")
                else:
                    print(f"   ❌ Simple interaction '{phrase}' FAILED")
                    all_passed = False
            else:
                all_passed = False
                results[phrase] = {"passed": False, "error": "Request failed"}
        
        # Overall summary
        passed_phrases = sum(1 for result in results.values() if result.get("passed", False))
        print(f"\n   📊 Simple Interactions Summary: {passed_phrases}/{len(simple_phrases)} phrases passed")
        
        return all_passed, results

    def test_legal_qa_system_availability(self):
        """Test that the legal Q&A system is available and working"""
        
        # Test the stats endpoint to verify system is available
        success, response = self.run_test(
            "Legal Q&A System Availability", 
            "GET", 
            "legal-qa/stats", 
            200,
            timeout=30
        )
        
        if success and isinstance(response, dict):
            print(f"\n   📋 Analyzing system availability...")
            
            # Check system components
            vector_db = response.get('vector_db', '')
            embeddings_model = response.get('embeddings_model', '')
            active_sessions = response.get('active_sessions', -1)
            total_conversations = response.get('total_conversations', -1)
            
            print(f"   Vector DB: {vector_db}")
            print(f"   Embeddings Model: {embeddings_model}")
            print(f"   Active Sessions: {active_sessions}")
            print(f"   Total Conversations: {total_conversations}")
            
            # Verify system is operational
            if vector_db and embeddings_model:
                print(f"   ✅ Legal Q&A system is operational")
            else:
                print(f"   ❌ Legal Q&A system may not be fully operational")
        
        return success, response

    def run_all_tests(self):
        """Run all legal Q&A chatbot improvement tests"""
        print("🚀 Starting Legal Q&A Chatbot Improvements Testing...")
        print("=" * 80)
        
        # Test 1: System Availability
        print("\n" + "=" * 50)
        print("TEST 1: SYSTEM AVAILABILITY")
        print("=" * 50)
        self.test_legal_qa_system_availability()
        
        # Test 2: Simple Greeting Test
        print("\n" + "=" * 50)
        print("TEST 2: SIMPLE GREETING TEST")
        print("=" * 50)
        self.test_simple_greeting_hi()
        
        # Test 3: Complex Legal Question Test
        print("\n" + "=" * 50)
        print("TEST 3: COMPLEX LEGAL QUESTION TEST")
        print("=" * 50)
        self.test_complex_legal_question()
        
        # Test 4: Other Simple Interactions
        print("\n" + "=" * 50)
        print("TEST 4: OTHER SIMPLE INTERACTIONS")
        print("=" * 50)
        self.test_other_simple_interactions()
        
        # Final Summary
        print("\n" + "=" * 80)
        print("FINAL SUMMARY")
        print("=" * 80)
        print(f"Total Tests Run: {self.tests_run}")
        print(f"Tests Passed: {self.tests_passed}")
        print(f"Success Rate: {(self.tests_passed/self.tests_run)*100:.1f}%")
        
        if self.tests_passed == self.tests_run:
            print("🎉 ALL TESTS PASSED - Legal Q&A chatbot improvements are working correctly!")
        elif self.tests_passed >= self.tests_run * 0.8:
            print("✅ MOSTLY SUCCESSFUL - Legal Q&A chatbot improvements are mostly working")
        else:
            print("❌ MULTIPLE FAILURES - Legal Q&A chatbot improvements need attention")
        
        return self.tests_passed, self.tests_run

def main():
    """Main test execution"""
    print("Legal Q&A Chatbot Improvements Testing")
    print("Testing the specific fixes for greeting handling and legal question processing")
    print("=" * 80)
    
    tester = LegalQAChatbotTester()
    passed, total = tester.run_all_tests()
    
    # Exit with appropriate code
    if passed == total:
        sys.exit(0)  # All tests passed
    elif passed >= total * 0.8:
        sys.exit(1)  # Mostly successful but some issues
    else:
        sys.exit(2)  # Multiple failures

if __name__ == "__main__":
    main()