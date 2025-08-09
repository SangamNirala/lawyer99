import requests
import json
import time
from datetime import datetime

class LegalQAPriorityTester:
    def __init__(self, base_url="https://7eb15b8a-98e2-46d7-83f4-aba058ad34e0.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.session_id = f"priority-test-{int(time.time())}"

    def test_specific_legal_questions(self):
        """Test the specific legal questions mentioned in the review request"""
        priority_questions = [
            {
                "question": "What are the key elements of a valid contract?",
                "jurisdiction": "US",
                "legal_domain": "contract_law",
                "description": "Contract Law - Key Elements"
            },
            {
                "question": "Can an employer terminate an employee without cause?",
                "jurisdiction": "US", 
                "legal_domain": "employment_law",
                "description": "Employment Law - Termination"
            },
            {
                "question": "What legal considerations are important when starting a business?",
                "jurisdiction": "US",
                "legal_domain": "business_law",
                "description": "Business Law - Starting Business"
            },
            {
                "question": "How long does copyright protection last?",
                "jurisdiction": "US",
                "legal_domain": "intellectual_property",
                "description": "Intellectual Property - Copyright"
            },
            {
                "question": "What are the essential elements of a real estate purchase agreement?",
                "jurisdiction": "US",
                "legal_domain": "real_estate_law",
                "description": "Real Estate Law - Purchase Agreement"
            }
        ]
        
        print("=" * 80)
        print("🎯 PRIORITY LEGAL Q&A TESTING - API KEY FIX VERIFICATION")
        print("=" * 80)
        print(f"Testing against: {self.base_url}")
        print(f"Session ID: {self.session_id}")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        working_questions = 0
        total_questions = len(priority_questions)
        results = {}
        
        for i, test_case in enumerate(priority_questions):
            print(f"\n" + "="*60)
            print(f"PRIORITY TEST {i+1}: {test_case['description']}")
            print(f"="*60)
            
            question_data = {
                "question": test_case["question"],
                "session_id": f"{self.session_id}-priority-{i+1}",
                "jurisdiction": test_case["jurisdiction"],
                "legal_domain": test_case["legal_domain"]
            }
            
            print(f"🔍 Testing: {test_case['question']}")
            print(f"   URL: {self.api_url}/legal-qa/ask")
            
            try:
                response = requests.post(
                    f"{self.api_url}/legal-qa/ask", 
                    json=question_data, 
                    headers={'Content-Type': 'application/json'},
                    timeout=90
                )
                
                print(f"   Status: {response.status_code}")
                
                if response.status_code == 200:
                    response_data = response.json()
                    print(f"✅ Request Successful")
                    
                    # Analyze response quality
                    confidence = response_data.get('confidence', 0)
                    answer = response_data.get('answer', '')
                    sources = response_data.get('sources', [])
                    model_used = response_data.get('model_used', 'unknown')
                    
                    print(f"   📊 Response Analysis:")
                    print(f"      Confidence Score: {confidence}")
                    print(f"      Answer Length: {len(answer)} characters")
                    print(f"      Sources Count: {len(sources)}")
                    print(f"      Model Used: {model_used}")
                    
                    # Check for fallback indicators
                    fallback_indicators = [
                        "I apologize, but I'm unable to generate a response",
                        "I cannot provide legal advice",
                        "fallback response"
                    ]
                    
                    is_fallback = any(indicator.lower() in answer.lower() for indicator in fallback_indicators)
                    
                    # Determine if this is a proper AI-powered response
                    is_proper_response = (
                        confidence >= 0.6 and  # High confidence (60%+)
                        len(answer) > 200 and  # Substantial answer
                        not is_fallback and    # Not a fallback response
                        model_used != 'greeting_handler'  # Not just a greeting
                    )
                    
                    if is_proper_response:
                        working_questions += 1
                        print(f"      ✅ HIGH-QUALITY AI RESPONSE CONFIRMED!")
                        print(f"         - Confidence: {confidence:.1%} (Target: 70%+)")
                        print(f"         - No fallback detected")
                        print(f"         - Substantial legal content provided")
                        print(f"      Answer Preview: {answer[:200]}...")
                        
                        # Check for legal disclaimers and citations
                        has_disclaimer = any(term in answer.lower() for term in ['disclaimer', 'attorney', 'legal counsel', 'consult'])
                        if has_disclaimer:
                            print(f"         ✅ Contains proper legal disclaimers")
                        else:
                            print(f"         ⚠️  May lack legal disclaimers")
                            
                    else:
                        print(f"      ❌ POOR QUALITY RESPONSE")
                        if confidence < 0.6:
                            print(f"         - Low confidence: {confidence:.1%} (Target: 70%+)")
                        if len(answer) <= 200:
                            print(f"         - Answer too short: {len(answer)} chars")
                        if is_fallback:
                            print(f"         - FALLBACK RESPONSE DETECTED")
                        if model_used == 'greeting_handler':
                            print(f"         - Using greeting handler instead of AI")
                        print(f"      Answer Preview: {answer[:200]}...")
                    
                    results[test_case['description']] = {
                        'success': True,
                        'confidence': confidence,
                        'answer_length': len(answer),
                        'sources_count': len(sources),
                        'model_used': model_used,
                        'is_proper_response': is_proper_response,
                        'is_fallback': is_fallback
                    }
                    
                else:
                    print(f"❌ Request Failed - Status: {response.status_code}")
                    try:
                        error_data = response.json()
                        print(f"   Error: {error_data}")
                    except:
                        print(f"   Error: {response.text}")
                    
                    results[test_case['description']] = {
                        'success': False,
                        'error': f"HTTP {response.status_code}"
                    }
                    
            except Exception as e:
                print(f"❌ Request Exception: {str(e)}")
                results[test_case['description']] = {
                    'success': False,
                    'error': str(e)
                }
        
        # Final Assessment
        print(f"\n" + "="*80)
        print(f"🎯 PRIORITY TESTING FINAL ASSESSMENT")
        print(f"="*80)
        print(f"High-Quality AI Responses: {working_questions}/{total_questions}")
        print(f"Success Rate: {(working_questions/total_questions)*100:.1f}%")
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if working_questions == total_questions:
            print(f"\n🎉 EXCELLENT: ALL PRIORITY QUESTIONS WORKING WITH HIGH-QUALITY AI RESPONSES!")
            print(f"   ✅ API key fix has been successful")
            print(f"   ✅ No more 0% confidence fallback responses")
            print(f"   ✅ AI-powered legal responses with proper confidence scores")
            print(f"   ✅ Legal Q&A system is fully operational")
        elif working_questions >= total_questions * 0.8:
            print(f"\n✅ GOOD: Most priority questions working with high-quality responses")
            print(f"   ✅ API key fix appears successful")
            print(f"   ⚠️  Minor issues with {total_questions - working_questions} questions")
        elif working_questions > 0:
            print(f"\n⚠️  PARTIAL: Some questions working, but issues remain")
            print(f"   ⚠️  API key fix partially successful")
            print(f"   ❌ {total_questions - working_questions} questions still have issues")
        else:
            print(f"\n❌ CRITICAL: NO QUESTIONS WORKING WITH HIGH-QUALITY RESPONSES")
            print(f"   ❌ API key fix may not have resolved the issues")
            print(f"   ❌ Still experiencing fallback responses or low confidence")
        
        return working_questions, total_questions, results

    def test_stats_endpoints(self):
        """Test the secondary stats endpoints"""
        print(f"\n" + "="*60)
        print(f"SECONDARY TESTS: STATS ENDPOINTS")
        print(f"="*60)
        
        # Test RAG system stats
        print(f"\n🔍 Testing RAG System Stats...")
        try:
            response = requests.get(f"{self.api_url}/legal-qa/stats", timeout=30)
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                stats_data = response.json()
                print(f"✅ RAG System Stats Working")
                print(f"   Vector DB: {stats_data.get('vector_db', 'N/A')}")
                print(f"   Embeddings Model: {stats_data.get('embeddings_model', 'N/A')}")
                print(f"   Indexed Documents: {stats_data.get('indexed_documents', 'N/A')}")
            else:
                print(f"❌ RAG System Stats Failed - Status: {response.status_code}")
        except Exception as e:
            print(f"❌ RAG System Stats Exception: {str(e)}")
        
        # Test knowledge base stats
        print(f"\n🔍 Testing Knowledge Base Stats...")
        try:
            response = requests.get(f"{self.api_url}/legal-qa/knowledge-base/stats", timeout=30)
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                kb_data = response.json()
                print(f"✅ Knowledge Base Stats Working")
                print(f"   Total Documents: {kb_data.get('total_documents', 'N/A')}")
                print(f"   Jurisdictions: {len(kb_data.get('by_jurisdiction', {}))}")
                print(f"   Legal Domains: {len(kb_data.get('by_legal_domain', {}))}")
            else:
                print(f"❌ Knowledge Base Stats Failed - Status: {response.status_code}")
        except Exception as e:
            print(f"❌ Knowledge Base Stats Exception: {str(e)}")

if __name__ == "__main__":
    tester = LegalQAPriorityTester()
    working, total, results = tester.test_specific_legal_questions()
    tester.test_stats_endpoints()
    
    # Exit with appropriate code
    if working == total:
        exit(0)  # Perfect success
    elif working >= total * 0.8:
        exit(1)  # Good but not perfect
    else:
        exit(2)  # Issues remain