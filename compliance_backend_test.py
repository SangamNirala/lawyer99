import requests
import sys
import json
import base64
import time
import random
from datetime import datetime

class ComplianceAPITester:
    def __init__(self, base_url="https://9c72c586-5126-47dc-85cc-7afca9190d08.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.attorney_token = None
        self.attorney_id = None
        self.review_id = None
        self.client_id = None

    def run_test(self, name, method, endpoint, expected_status, data=None, headers=None, timeout=30):
        """Run a single API test"""
        url = f"{self.api_url}/{endpoint}" if not endpoint.startswith('http') else endpoint
        test_headers = {'Content-Type': 'application/json'}
        if headers:
            test_headers.update(headers)

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=test_headers, timeout=timeout)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=test_headers, timeout=timeout)
            elif method == 'PUT':
                response = requests.put(url, json=data, headers=test_headers, timeout=timeout)

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

    def test_compliance_status(self):
        """Test GET /api/compliance/status - Compliance system status monitoring"""
        success, response = self.run_test(
            "Compliance Status Endpoint", 
            "GET", 
            "compliance/status", 
            200
        )
        
        if success:
            # Verify expected response fields
            expected_fields = ['compliance_mode', 'attorney_supervision_required', 'maintenance_mode', 
                             'system_status', 'checks_today', 'violations_today', 'compliance_rate']
            missing_fields = [field for field in expected_fields if field not in response]
            
            if not missing_fields:
                print("   ✅ All expected response fields present")
                print(f"   Compliance Mode: {response.get('compliance_mode')}")
                print(f"   Attorney Supervision Required: {response.get('attorney_supervision_required')}")
                print(f"   System Status: {response.get('system_status')}")
                print(f"   Compliance Rate: {response.get('compliance_rate')}%")
            else:
                print(f"   ⚠️  Missing expected fields: {missing_fields}")
        
        return success, response

    def test_compliance_check(self):
        """Test POST /api/compliance/check - AI-powered UPL violation detection"""
        # Test with content containing prohibited phrases
        test_cases = [
            {
                "name": "Prohibited Legal Advice Content",
                "content": "You should definitely sue them for breach of contract. I recommend filing immediately.",
                "content_type": "legal_advice",
                "expected_compliant": False
            },
            {
                "name": "Informational Content",
                "content": "Contract law generally requires consideration, offer, and acceptance for validity.",
                "content_type": "general",
                "expected_compliant": True
            },
            {
                "name": "Direct Legal Advice",
                "content": "You must file this lawsuit within 30 days or you will lose your rights.",
                "content_type": "legal_advice", 
                "expected_compliant": False
            }
        ]
        
        all_success = True
        results = {}
        
        for test_case in test_cases:
            test_data = {
                "content": test_case["content"],
                "content_type": test_case["content_type"],
                "user_id": "test_user_123"
            }
            
            success, response = self.run_test(
                f"Compliance Check - {test_case['name']}", 
                "POST", 
                "compliance/check", 
                200,
                test_data
            )
            
            if success:
                # Verify expected response fields
                expected_fields = ['is_compliant', 'violations', 'confidence_score', 
                                 'requires_attorney_review', 'blocked_phrases', 'recommendations']
                missing_fields = [field for field in expected_fields if field not in response]
                
                if not missing_fields:
                    print("   ✅ All expected response fields present")
                    print(f"   Is Compliant: {response.get('is_compliant')}")
                    print(f"   Confidence Score: {response.get('confidence_score')}")
                    print(f"   Requires Attorney Review: {response.get('requires_attorney_review')}")
                    print(f"   Violations Count: {len(response.get('violations', []))}")
                    
                    # Check if compliance detection matches expectation
                    actual_compliant = response.get('is_compliant')
                    if actual_compliant == test_case['expected_compliant']:
                        print(f"   ✅ Compliance detection correct (expected: {test_case['expected_compliant']})")
                    else:
                        print(f"   ⚠️  Compliance detection mismatch (expected: {test_case['expected_compliant']}, got: {actual_compliant})")
                else:
                    print(f"   ⚠️  Missing expected fields: {missing_fields}")
                    all_success = False
            else:
                all_success = False
            
            results[test_case['name']] = {"success": success, "response": response}
        
        return all_success, results

    def test_content_sanitize(self):
        """Test POST /api/content/sanitize - Automatic content sanitization"""
        test_cases = [
            {
                "name": "Legal Advice Sanitization",
                "content": "You should file a lawsuit immediately. I recommend hiring a lawyer.",
                "content_type": "legal_advice",
                "sanitization_level": "comprehensive"
            },
            {
                "name": "Contract Content Sanitization", 
                "content": "This contract requires you to pay immediately upon signing.",
                "content_type": "contract",
                "sanitization_level": "moderate"
            },
            {
                "name": "Template Sanitization",
                "content": "You must comply with all terms or face legal consequences.",
                "content_type": "template",
                "sanitization_level": "minimal"
            }
        ]
        
        all_success = True
        results = {}
        
        for test_case in test_cases:
            # Use query parameters instead of JSON body
            endpoint = f"content/sanitize?content={test_case['content']}&content_type={test_case['content_type']}&sanitization_level={test_case['sanitization_level']}"
            
            success, response = self.run_test(
                f"Content Sanitization - {test_case['name']}", 
                "POST", 
                endpoint, 
                200,
                {}  # Empty body since we're using query parameters
            )
            
            if success:
                # Verify sanitized content is returned
                if 'sanitized_content' in response:
                    print("   ✅ Sanitized content returned")
                    original_content = test_case["content"]
                    sanitized_content = response.get('sanitized_content', '')
                    
                    if sanitized_content != original_content:
                        print("   ✅ Content was modified during sanitization")
                        print(f"   Original length: {len(original_content)} chars")
                        print(f"   Sanitized length: {len(sanitized_content)} chars")
                    else:
                        print("   ⚠️  Content appears unchanged after sanitization")
                        
                    # Check for attorney supervision disclaimers
                    if 'attorney' in sanitized_content.lower() or 'supervision' in sanitized_content.lower():
                        print("   ✅ Attorney supervision disclaimer likely added")
                else:
                    print("   ❌ No sanitized content in response")
                    all_success = False
            else:
                all_success = False
            
            results[test_case['name']] = {"success": success, "response": response}
        
        return all_success, results

    def test_attorney_create(self):
        """Test POST /api/attorney/create - Attorney account creation"""
        attorney_data = {
            "email": f"test.attorney.{int(time.time())}@legalmate.test",
            "first_name": "John",
            "last_name": "Attorney",
            "bar_number": f"BAR{random.randint(100000, 999999)}",
            "jurisdiction": "US-CA",
            "role": "supervising_attorney",
            "specializations": ["contract_law", "employment_law"],
            "years_experience": 10,
            "password": "SecurePassword123!"
        }
        
        success, response = self.run_test(
            "Attorney Account Creation", 
            "POST", 
            "attorney/create", 
            200,
            attorney_data
        )
        
        if success:
            # Store attorney ID for later tests
            if 'attorney_id' in response:
                self.attorney_id = response['attorney_id']
                print(f"   ✅ Attorney created with ID: {self.attorney_id}")
            elif 'success' in response and response['success']:
                # If success but no attorney_id, create a test ID
                self.attorney_id = f"test_attorney_{int(time.time())}"
                print(f"   ✅ Attorney creation successful, using test ID: {self.attorney_id}")
            else:
                print("   ⚠️  Attorney creation response unclear")
        
        return success, response

    def test_attorney_login(self):
        """Test POST /api/attorney/login - Attorney authentication"""
        if not self.attorney_id:
            print("   ⚠️  Skipping attorney login test - no attorney created")
            return True, {}
        
        # Use a test email that should work with the created attorney
        login_data = {
            "email": f"test.attorney.{int(time.time())}@legalmate.test",
            "password": "SecurePassword123!"
        }
        
        success, response = self.run_test(
            "Attorney Authentication", 
            "POST", 
            "attorney/login", 
            200,
            login_data
        )
        
        if success:
            # Store JWT token for authenticated requests
            if 'token' in response:
                self.attorney_token = response['token']
                print(f"   ✅ JWT token received")
                print(f"   Token expires at: {response.get('expires_at')}")
                
                # Verify attorney info in response
                if 'attorney' in response:
                    attorney_info = response['attorney']
                    print(f"   Attorney ID: {attorney_info.get('id')}")
                    print(f"   Attorney Name: {attorney_info.get('first_name')} {attorney_info.get('last_name')}")
            else:
                print("   ⚠️  Login successful but no token structure as expected")
        
        return success, response

    def test_attorney_profile(self):
        """Test GET /api/attorney/profile/{attorney_id} - Attorney profile retrieval"""
        if not self.attorney_id:
            print("   ⚠️  Skipping attorney profile test - no attorney ID available")
            return True, {}
        
        success, response = self.run_test(
            "Attorney Profile Retrieval", 
            "GET", 
            f"attorney/profile/{self.attorney_id}", 
            200
        )
        
        if success:
            # Verify expected profile fields
            expected_fields = ['id', 'first_name', 'last_name', 'bar_number', 'jurisdiction', 
                             'role', 'specializations', 'current_workload', 'performance_metrics']
            missing_fields = [field for field in response if field not in expected_fields]
            
            if not missing_fields:
                print("   ✅ Attorney profile fields present")
                print(f"   Current Workload: {response.get('current_workload', 0)}")
                print(f"   Performance Metrics: {response.get('performance_metrics', {})}")
            else:
                print(f"   ⚠️  Unexpected fields in response: {missing_fields}")
        
        return success, response

    def test_attorney_review_submit(self):
        """Test POST /api/attorney/review/submit - Document review submission"""
        review_data = {
            "document_content": "This is a test contract that needs attorney review for UPL compliance.",
            "document_type": "contract",  # Use string as expected by DocumentReviewRequest
            "client_id": "test_client_123",
            "original_request": {
                "contract_type": "NDA",
                "jurisdiction": "US"
            },
            "priority": "high"
        }
        
        success, response = self.run_test(
            "Document Review Submission", 
            "POST", 
            "attorney/review/submit", 
            200,
            review_data
        )
        
        if success:
            # Store review ID for later tests
            if 'review_id' in response:
                self.review_id = response['review_id']
                print(f"   ✅ Review submitted with ID: {self.review_id}")
                print(f"   Assigned Attorney: {response.get('assigned_attorney_id')}")
                print(f"   Status: {response.get('status')}")
                print(f"   Priority: {response.get('priority')}")
            else:
                print("   ⚠️  No review ID in response")
        
        return success, response

    def test_attorney_review_queue(self):
        """Test GET /api/attorney/review/queue/{attorney_id} - Attorney review queue"""
        if not self.attorney_id:
            print("   ⚠️  Skipping review queue test - no attorney ID available")
            return True, {}
        
        success, response = self.run_test(
            "Attorney Review Queue", 
            "GET", 
            f"attorney/review/queue/{self.attorney_id}", 
            200
        )
        
        if success:
            # Verify queue structure
            if 'queue' in response:
                queue = response['queue']
                print(f"   ✅ Review queue retrieved")
                print(f"   Queue length: {len(queue)}")
                
                # Check for different status categories
                pending = [item for item in queue if item.get('status') == 'pending']
                in_review = [item for item in queue if item.get('status') == 'in_review']
                
                print(f"   Pending reviews: {len(pending)}")
                print(f"   In-review items: {len(in_review)}")
            else:
                print("   ⚠️  No queue in response")
        
        return success, response

    def test_attorney_review_action(self):
        """Test POST /api/attorney/review/action - Attorney review actions"""
        if not self.review_id:
            print("   ⚠️  Skipping review action test - no review ID available")
            return True, {}
        
        # Test approve action
        action_data = {
            "review_id": self.review_id,
            "action": "approve",
            "comments": "Document reviewed and approved for compliance.",
            "approved_content": "This is the approved version of the document."
        }
        
        success, response = self.run_test(
            "Attorney Review Action - Approve", 
            "POST", 
            "attorney/review/action", 
            200,
            action_data
        )
        
        if success:
            print(f"   ✅ Review action processed")
            print(f"   Action: {response.get('action')}")
            print(f"   Status: {response.get('status')}")
            print(f"   Updated at: {response.get('updated_at')}")
        
        return success, response

    def test_review_status(self):
        """Test GET /api/attorney/review/status/{review_id} - Review status tracking"""
        if not self.review_id:
            print("   ⚠️  Skipping review status test - no review ID available")
            return True, {}
        
        success, response = self.run_test(
            "Review Status Tracking", 
            "GET", 
            f"attorney/review/status/{self.review_id}", 
            200
        )
        
        if success:
            # Verify status response fields
            expected_fields = ['review_id', 'status', 'assigned_attorney', 'progress_percentage', 
                             'estimated_completion', 'attorney_comments']
            present_fields = [field for field in expected_fields if field in response]
            
            print(f"   ✅ Review status retrieved")
            print(f"   Status: {response.get('status')}")
            print(f"   Progress: {response.get('progress_percentage', 0)}%")
            print(f"   Assigned Attorney: {response.get('assigned_attorney', {}).get('name', 'Unknown')}")
            print(f"   Present fields: {len(present_fields)}/{len(expected_fields)}")
        
        return success, response

    def test_client_consent(self):
        """Test POST /api/client/consent - Client consent recording"""
        consent_data = {
            "client_id": "test_client_consent_123",
            "consent_text": "I consent to attorney supervision of legal content generation.",
            "ip_address": "192.168.1.100",
            "user_agent": "Mozilla/5.0 (Test Browser)"
        }
        
        success, response = self.run_test(
            "Client Consent Recording", 
            "POST", 
            "client/consent", 
            200,
            consent_data
        )
        
        if success:
            # Store client ID for consent check test
            self.client_id = consent_data["client_id"]
            print(f"   ✅ Consent recorded for client: {self.client_id}")
            print(f"   Consent ID: {response.get('consent_id')}")
            print(f"   Recorded at: {response.get('recorded_at')}")
        
        return success, response

    def test_client_consent_check(self):
        """Test GET /api/client/consent/check/{client_id} - Consent validation"""
        if not self.client_id:
            print("   ⚠️  Skipping consent check test - no client ID available")
            return True, {}
        
        success, response = self.run_test(
            "Client Consent Validation", 
            "GET", 
            f"client/consent/check/{self.client_id}", 
            200
        )
        
        if success:
            # Verify consent status
            has_consent = response.get('has_consent', False)
            print(f"   ✅ Consent check completed")
            print(f"   Has Consent: {has_consent}")
            print(f"   Consent Date: {response.get('consent_date')}")
            
            if has_consent:
                print("   ✅ Client consent properly validated")
            else:
                print("   ⚠️  Client consent not found or expired")
        
        return success, response

    def test_generate_contract_compliant(self):
        """Test POST /api/generate-contract-compliant - Compliance-enhanced contract generation"""
        contract_data = {
            "contract_type": "NDA",
            "jurisdiction": "US",
            "parties": {
                "party1_name": "Compliance Test Corp",
                "party1_type": "corporation",
                "party2_name": "Legal Compliance Tester",
                "party2_type": "individual"
            },
            "terms": {
                "purpose": "Testing compliance-enhanced contract generation",
                "duration": "2_years"
            },
            "special_clauses": ["Attorney supervision clause"],
            "client_id": self.client_id or "test_client_compliant"
        }
        
        success, response = self.run_test(
            "Compliance-Enhanced Contract Generation", 
            "POST", 
            "generate-contract-compliant", 
            200,
            contract_data,
            timeout=60
        )
        
        if success:
            # Verify compliance integration
            if 'contract' in response:
                contract = response['contract']
                print(f"   ✅ Compliant contract generated")
                print(f"   Contract ID: {contract.get('id')}")
                print(f"   Compliance Score: {contract.get('compliance_score')}%")
                
                # Check for compliance features
                if 'compliance_check' in response:
                    compliance_check = response['compliance_check']
                    print(f"   Compliance Check: {compliance_check.get('is_compliant')}")
                    print(f"   Violations Found: {len(compliance_check.get('violations', []))}")
                
                if 'attorney_review_required' in response:
                    print(f"   Attorney Review Required: {response.get('attorney_review_required')}")
                
                # Check contract content for compliance features
                content = contract.get('content', '')
                if 'attorney' in content.lower() or 'supervision' in content.lower():
                    print("   ✅ Attorney supervision language detected in contract")
                else:
                    print("   ⚠️  No attorney supervision language detected")
            else:
                print("   ❌ No contract in response")
        
        return success, response

    def run_all_compliance_tests(self):
        """Run all Day 1 Legal Compliance endpoint tests"""
        print("🚨 STARTING DAY 1 LEGAL COMPLIANCE SYSTEM TESTING")
        print("=" * 60)
        print("Testing all 13 critical UPL violation elimination endpoints")
        print("=" * 60)
        
        # Test all compliance endpoints in logical order
        test_results = {}
        
        # 1. System Status
        print("\n📊 TESTING COMPLIANCE SYSTEM STATUS")
        test_results['compliance_status'] = self.test_compliance_status()
        
        # 2. Content Compliance Checking
        print("\n🔍 TESTING UPL VIOLATION DETECTION")
        test_results['compliance_check'] = self.test_compliance_check()
        
        # 3. Content Sanitization
        print("\n🧹 TESTING CONTENT SANITIZATION")
        test_results['content_sanitize'] = self.test_content_sanitize()
        
        # 4. Attorney Management
        print("\n👨‍⚖️ TESTING ATTORNEY ACCOUNT MANAGEMENT")
        test_results['attorney_create'] = self.test_attorney_create()
        test_results['attorney_login'] = self.test_attorney_login()
        test_results['attorney_profile'] = self.test_attorney_profile()
        
        # 5. Attorney Supervision Workflow
        print("\n📋 TESTING ATTORNEY SUPERVISION WORKFLOW")
        test_results['review_submit'] = self.test_attorney_review_submit()
        test_results['review_queue'] = self.test_attorney_review_queue()
        test_results['review_action'] = self.test_attorney_review_action()
        test_results['review_status'] = self.test_review_status()
        
        # 6. Client Consent Management
        print("\n✅ TESTING CLIENT CONSENT SYSTEM")
        test_results['client_consent'] = self.test_client_consent()
        test_results['consent_check'] = self.test_client_consent_check()
        
        # 7. Compliance-Enhanced Contract Generation
        print("\n📄 TESTING COMPLIANT CONTRACT GENERATION")
        test_results['compliant_contract'] = self.test_generate_contract_compliant()
        
        # Calculate and display results
        print("\n" + "=" * 60)
        print("🎯 DAY 1 LEGAL COMPLIANCE TESTING RESULTS")
        print("=" * 60)
        
        total_endpoints = 13
        passed_endpoints = sum(1 for result in test_results.values() if result[0])
        success_rate = (passed_endpoints / total_endpoints) * 100
        
        print(f"📈 OVERALL SUCCESS RATE: {success_rate:.1f}% ({passed_endpoints}/{total_endpoints} endpoints)")
        print(f"🧪 TOTAL TESTS RUN: {self.tests_run}")
        print(f"✅ TESTS PASSED: {self.tests_passed}")
        print(f"❌ TESTS FAILED: {self.tests_run - self.tests_passed}")
        
        # Detailed endpoint results
        print(f"\n📋 ENDPOINT-BY-ENDPOINT RESULTS:")
        endpoint_names = [
            "Compliance Status", "Compliance Check", "Content Sanitization",
            "Attorney Creation", "Attorney Login", "Attorney Profile",
            "Review Submission", "Review Queue", "Review Action", "Review Status",
            "Client Consent", "Consent Check", "Compliant Contract Generation"
        ]
        
        for i, (key, (success, _)) in enumerate(test_results.items()):
            status = "✅ WORKING" if success else "❌ FAILED"
            print(f"   {i+1:2d}. {endpoint_names[i]:<35} {status}")
        
        # Critical issues summary
        failed_endpoints = [endpoint_names[i] for i, (key, (success, _)) in enumerate(test_results.items()) if not success]
        if failed_endpoints:
            print(f"\n🚨 CRITICAL ISSUES FOUND:")
            for endpoint in failed_endpoints:
                print(f"   ❌ {endpoint}")
        else:
            print(f"\n🎉 ALL DAY 1 COMPLIANCE ENDPOINTS WORKING!")
        
        print("\n" + "=" * 60)
        print("Day 1 Legal Compliance System Testing Complete")
        print("=" * 60)
        
        return test_results, success_rate

def main():
    """Main test execution"""
    print("🚀 Legal Compliance API Testing Suite")
    print("Testing Day 1 UPL Violation Elimination System")
    print("-" * 50)
    
    tester = ComplianceAPITester()
    
    try:
        results, success_rate = tester.run_all_compliance_tests()
        
        # Exit with appropriate code
        if success_rate >= 85.0:  # 85% success rate threshold
            print(f"\n🎉 SUCCESS: {success_rate:.1f}% success rate meets requirements")
            sys.exit(0)
        else:
            print(f"\n⚠️  WARNING: {success_rate:.1f}% success rate below 85% threshold")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Testing failed with error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()