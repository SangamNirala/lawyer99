import requests
import sys
import json
import random
import time
from datetime import datetime

class VoiceAgentTester:
    def __init__(self, base_url="https://027a3f35-a0bc-40cd-ad79-483d46c6126e.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.voice_session_ids = []

    def generate_voice_session_id(self):
        """Generate voice session ID with expected format"""
        timestamp = int(time.time())
        random_suffix = random.randint(1000, 9999)
        return f"voice_session_{timestamp}_{random_suffix}"

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

    def test_voice_agent_legal_qa_basic(self):
        """Test basic Voice Agent legal Q&A functionality"""
        print(f"\n🎤 Testing Voice Agent Basic Legal Q&A...")
        
        test_data = {
            "question": "What are the essential elements of a valid contract under US law?",
            "jurisdiction": "US",
            "legal_domain": "contract_law",
            "client_type": "voice_assistant"
        }
        
        success, response = self.run_test(
            "Voice Agent Basic Legal Q&A",
            "POST",
            "legal-qa/voice-ask",
            200,
            test_data
        )
        
        if success and response:
            # Verify voice session format
            voice_session_id = response.get("voice_session_id", "")
            if voice_session_id.startswith("voice_session_"):
                print(f"   ✅ Voice session ID format correct: {voice_session_id}")
                self.voice_session_ids.append(voice_session_id)
            else:
                print(f"   ❌ Invalid voice session ID format: {voice_session_id}")
                
            # Verify response structure
            required_fields = ["answer", "confidence", "sources", "voice_session_id", "is_voice_session"]
            missing_fields = [field for field in required_fields if field not in response]
            if not missing_fields:
                print(f"   ✅ All required response fields present")
            else:
                print(f"   ❌ Missing fields: {missing_fields}")
                
            # Verify it's marked as voice session
            if response.get("is_voice_session") == True:
                print(f"   ✅ Correctly marked as voice session")
            else:
                print(f"   ❌ Not marked as voice session")
                
        return success, response

    def test_voice_agent_multi_turn_conversation(self):
        """Test multi-turn conversation with voice session continuity"""
        print(f"\n🔄 Testing Voice Agent Multi-turn Conversation...")
        
        # First question
        first_question = {
            "question": "What is the difference between an offer and an invitation to treat?",
            "jurisdiction": "US",
            "legal_domain": "contract_law",
            "client_type": "voice_assistant"
        }
        
        success1, response1 = self.run_test(
            "Voice Agent Multi-turn - First Question",
            "POST",
            "legal-qa/voice-ask",
            200,
            first_question
        )
        
        if not success1:
            return False, {}
            
        voice_session_id = response1.get("voice_session_id")
        
        # Second question using same voice session
        second_question = {
            "question": "Can you give me an example of each from the previous answer?",
            "voice_session_id": voice_session_id,
            "jurisdiction": "US",
            "legal_domain": "contract_law",
            "client_type": "voice_assistant"
        }
        
        success2, response2 = self.run_test(
            "Voice Agent Multi-turn - Follow-up Question",
            "POST",
            "legal-qa/voice-ask",
            200,
            second_question
        )
        
        if success2 and response2:
            # Verify same voice session ID
            if response2.get("voice_session_id") == voice_session_id:
                print(f"   ✅ Voice session continuity maintained: {voice_session_id}")
            else:
                print(f"   ❌ Voice session continuity broken")
                
        return success2, response2

    def test_voice_agent_jurisdiction_support(self):
        """Test Voice Agent support for different jurisdictions"""
        print(f"\n🌍 Testing Voice Agent Jurisdiction Support...")
        
        jurisdictions = ["US", "UK", "CA", "AU", "EU", "IN"]
        jurisdiction_results = []
        
        for jurisdiction in jurisdictions:
            test_data = {
                "question": f"What are the key employment law requirements in {jurisdiction}?",
                "jurisdiction": jurisdiction,
                "legal_domain": "employment_labor_law",
                "client_type": "voice_assistant"
            }
            
            success, response = self.run_test(
                f"Voice Agent - {jurisdiction} Jurisdiction",
                "POST",
                "legal-qa/voice-ask",
                200,
                test_data
            )
            
            jurisdiction_results.append({
                "jurisdiction": jurisdiction,
                "success": success,
                "voice_session_id": response.get("voice_session_id") if success else None
            })
            
        # Summary
        successful_jurisdictions = [r["jurisdiction"] for r in jurisdiction_results if r["success"]]
        print(f"   ✅ Successful jurisdictions: {successful_jurisdictions}")
        print(f"   📊 Success rate: {len(successful_jurisdictions)}/{len(jurisdictions)}")
        
        return len(successful_jurisdictions) > 0, jurisdiction_results

    def test_voice_agent_legal_domain_filtering(self):
        """Test Voice Agent legal domain filtering"""
        print(f"\n⚖️ Testing Voice Agent Legal Domain Filtering...")
        
        legal_domains = [
            "contract_law",
            "employment_labor_law", 
            "intellectual_property",
            "corporate_law"
        ]
        
        domain_results = []
        
        for domain in legal_domains:
            test_data = {
                "question": f"What are the main principles of {domain.replace('_', ' ')}?",
                "jurisdiction": "US",
                "legal_domain": domain,
                "client_type": "voice_assistant"
            }
            
            success, response = self.run_test(
                f"Voice Agent - {domain}",
                "POST",
                "legal-qa/voice-ask",
                200,
                test_data
            )
            
            domain_results.append({
                "domain": domain,
                "success": success,
                "confidence": response.get("confidence") if success else 0
            })
            
        # Summary
        successful_domains = [r["domain"] for r in domain_results if r["success"]]
        avg_confidence = sum([r["confidence"] for r in domain_results if r["success"]]) / len(successful_domains) if successful_domains else 0
        
        print(f"   ✅ Successful domains: {successful_domains}")
        print(f"   📊 Success rate: {len(successful_domains)}/{len(legal_domains)}")
        print(f"   🎯 Average confidence: {avg_confidence:.2f}")
        
        return len(successful_domains) > 0, domain_results

    def test_voice_agent_session_management(self):
        """Test Voice Agent session management and format validation"""
        print(f"\n🗃️ Testing Voice Agent Session Management...")
        
        # Test 1: Auto-generation of voice session ID
        test_data = {
            "question": "What constitutes a breach of contract?",
            "jurisdiction": "US",
            "legal_domain": "contract_law",
            "client_type": "voice_assistant"
        }
        
        success, response = self.run_test(
            "Voice Session Auto-generation",
            "POST",
            "legal-qa/voice-ask",
            200,
            test_data
        )
        
        if not success:
            return False, {}
            
        auto_voice_session = response.get("voice_session_id")
        
        # Test 2: Using pre-generated voice session ID
        manual_voice_session = self.generate_voice_session_id()
        test_data_manual = {
            "question": "What are the remedies for breach of contract?",
            "voice_session_id": manual_voice_session,
            "jurisdiction": "US", 
            "legal_domain": "contract_law",
            "client_type": "voice_assistant"
        }
        
        success2, response2 = self.run_test(
            "Voice Session Manual ID",
            "POST",
            "legal-qa/voice-ask",
            200,
            test_data_manual
        )
        
        if success2:
            returned_session = response2.get("voice_session_id")
            if returned_session == manual_voice_session:
                print(f"   ✅ Manual voice session ID preserved: {manual_voice_session}")
            else:
                print(f"   ❌ Manual voice session ID not preserved")
                
        # Test 3: Voice session status endpoint
        if auto_voice_session:
            success3, response3 = self.run_test(
                "Voice Session Status Check",
                "GET",
                f"legal-qa/voice-session/{auto_voice_session}/status",
                200
            )
            
            if success3 and response3:
                if response3.get("session_format_valid") == True:
                    print(f"   ✅ Voice session format validation working")
                else:
                    print(f"   ❌ Voice session format validation failed")
                    
        return success and success2, {"auto_session": auto_voice_session, "manual_session": manual_voice_session}

    def test_voice_agent_error_handling(self):
        """Test Voice Agent error handling with invalid parameters"""
        print(f"\n🚨 Testing Voice Agent Error Handling...")
        
        # Test 1: Invalid voice session ID format
        invalid_session_data = {
            "question": "What is consideration in contract law?",
            "voice_session_id": "invalid_session_format",
            "jurisdiction": "US",
            "legal_domain": "contract_law",
            "client_type": "voice_assistant"
        }
        
        success1, response1 = self.run_test(
            "Voice Agent - Invalid Session Format",
            "POST",
            "legal-qa/voice-ask",
            200,  # Should succeed but generate new voice session ID
            invalid_session_data
        )
        
        if success1:
            new_session = response1.get("voice_session_id")
            if new_session and new_session.startswith("voice_session_"):
                print(f"   ✅ Invalid session ID handled correctly, new ID generated: {new_session}")
            else:
                print(f"   ❌ Invalid session ID not handled properly")
                
        # Test 2: Invalid voice session status check
        success2, response2 = self.run_test(
            "Voice Session Status - Invalid Format",
            "GET",
            "legal-qa/voice-session/invalid_format/status",
            400  # Should return error for invalid format
        )
        
        # Test 3: Missing required fields
        incomplete_data = {
            "jurisdiction": "US",
            "client_type": "voice_assistant"
            # Missing 'question' field
        }
        
        success3, response3 = self.run_test(
            "Voice Agent - Missing Question",
            "POST",
            "legal-qa/voice-ask",
            422,  # Validation error expected
            incomplete_data
        )
        
        error_tests_passed = 0
        if success1: error_tests_passed += 1
        if success2: error_tests_passed += 1  
        if success3: error_tests_passed += 1
        
        print(f"   📊 Error handling tests passed: {error_tests_passed}/3")
        
        return error_tests_passed >= 2, {"error_tests_passed": error_tests_passed}

    def test_voice_agent_response_structure(self):
        """Test Voice Agent response structure validation"""
        print(f"\n📋 Testing Voice Agent Response Structure...")
        
        test_data = {
            "question": "What are the elements of negligence in tort law?",
            "jurisdiction": "US",
            "legal_domain": "tort_law",
            "client_type": "voice_assistant"
        }
        
        success, response = self.run_test(
            "Voice Agent Response Structure",
            "POST",
            "legal-qa/voice-ask",
            200,
            test_data
        )
        
        if not success:
            return False, {}
            
        # Verify VoiceAgentResponse structure
        required_fields = [
            "answer", "confidence", "sources", "voice_session_id",
            "retrieved_documents", "timestamp", "is_voice_session", "session_metadata"
        ]
        
        structure_tests = []
        
        for field in required_fields:
            if field in response:
                print(f"   ✅ {field}: Present")
                structure_tests.append(True)
            else:
                print(f"   ❌ {field}: Missing")
                structure_tests.append(False)
                
        # Verify specific field types and values
        if "confidence" in response:
            confidence = response["confidence"]
            if isinstance(confidence, (int, float)) and 0 <= confidence <= 1:
                print(f"   ✅ Confidence in valid range: {confidence}")
            else:
                print(f"   ❌ Confidence out of range: {confidence}")
                
        if "is_voice_session" in response:
            if response["is_voice_session"] == True:
                print(f"   ✅ is_voice_session correctly set to True")
            else:
                print(f"   ❌ is_voice_session not set correctly")
                
        if "session_metadata" in response:
            metadata = response["session_metadata"]
            if isinstance(metadata, dict) and "client_type" in metadata:
                print(f"   ✅ Session metadata structure correct")
            else:
                print(f"   ❌ Session metadata structure incorrect")
                
        structure_score = sum(structure_tests) / len(structure_tests)
        print(f"   📊 Structure validation score: {structure_score:.2%}")
        
        return structure_score >= 0.8, {"structure_score": structure_score}

    def test_extended_legal_qa_with_voice_flag(self):
        """Test extended legal-qa/ask endpoint with is_voice flag"""
        print(f"\n🔄 Testing Extended Legal-QA Endpoint with Voice Flag...")
        
        # Test with is_voice=true
        test_data = {
            "question": "What is the statute of limitations for personal injury claims?",
            "is_voice": True,
            "jurisdiction": "US",
            "legal_domain": "tort_law"
        }
        
        success, response = self.run_test(
            "Legal-QA with Voice Flag",
            "POST",
            "legal-qa/ask",
            200,
            test_data
        )
        
        if success and response:
            session_id = response.get("session_id")
            is_voice_session = response.get("is_voice_session")
            
            if session_id and session_id.startswith("voice_session_"):
                print(f"   ✅ Voice session ID generated: {session_id}")
            else:
                print(f"   ❌ Voice session ID not generated properly")
                
            if is_voice_session == True:
                print(f"   ✅ is_voice_session flag set correctly")
            else:
                print(f"   ❌ is_voice_session flag not set")
                
        return success, response

    def run_all_voice_agent_tests(self):
        """Run comprehensive Voice Agent test suite"""
        print("🎤" + "="*70)
        print("🎤 VOICE AGENT BACKEND INTEGRATION TESTING")
        print("🎤" + "="*70)
        
        test_methods = [
            self.test_voice_agent_legal_qa_basic,
            self.test_voice_agent_multi_turn_conversation,
            self.test_voice_agent_jurisdiction_support,
            self.test_voice_agent_legal_domain_filtering,
            self.test_voice_agent_session_management,
            self.test_voice_agent_error_handling,
            self.test_voice_agent_response_structure,
            self.test_extended_legal_qa_with_voice_flag
        ]
        
        test_results = []
        
        for test_method in test_methods:
            try:
                success, result = test_method()
                test_results.append({
                    "test": test_method.__name__,
                    "success": success,
                    "result": result
                })
            except Exception as e:
                print(f"❌ Error in {test_method.__name__}: {e}")
                test_results.append({
                    "test": test_method.__name__,
                    "success": False,
                    "error": str(e)
                })
        
        # Final Summary
        print("\n" + "="*70)
        print("🎤 VOICE AGENT TESTING SUMMARY")
        print("="*70)
        
        passed_tests = [r for r in test_results if r["success"]]
        
        print(f"📊 Tests Run: {self.tests_run}")
        print(f"✅ Tests Passed: {self.tests_passed}")
        print(f"❌ Tests Failed: {self.tests_run - self.tests_passed}")
        print(f"🎯 Success Rate: {(self.tests_passed/self.tests_run)*100:.1f}%")
        
        print(f"\n🎤 Voice Agent Test Methods:")
        for result in test_results:
            status = "✅" if result["success"] else "❌"
            print(f"   {status} {result['test']}")
            
        print(f"\n🗃️ Generated Voice Sessions: {len(self.voice_session_ids)}")
        for session_id in self.voice_session_ids:
            print(f"   🎤 {session_id}")
            
        return self.tests_passed, self.tests_run, test_results

def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == "voice":
            tester = VoiceAgentTester()
            tester.run_all_voice_agent_tests()
            return
    
    print("Usage: python voice_agent_test.py voice")
    print("  voice  - Run Voice Agent backend integration tests")

if __name__ == "__main__":
    main()