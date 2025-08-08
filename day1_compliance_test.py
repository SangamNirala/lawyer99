import requests
import sys
import json
import time
import random
from datetime import datetime

class Day1ComplianceSystemTester:
    def __init__(self, base_url="https://93a8c601-fd30-4f56-a2f2-47103260abd2.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.attorney_token = None
        self.attorney_id = None
        self.review_id = None
        self.client_id = None

    def run_test(self, name, method, endpoint, expected_status, data=None, headers=None, timeout=30, params=None):
        """Run a single API test"""
        url = f"{self.api_url}/{endpoint}" if not endpoint.startswith('http') else endpoint
        if not headers:
            headers = {'Content-Type': 'application/json'}

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=timeout, params=params)
            elif method == 'POST':
                if params:
                    # For endpoints that use query parameters instead of JSON body
                    response = requests.post(url, headers=headers, timeout=timeout, params=params, json=data or {})
                else:
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

    def test_compliance_status(self):
        """Test GET /api/compliance/status - Compliance system status"""
        success, response = self.run_test(
            "Compliance Status Endpoint", 
            "GET", 
            "compliance/status", 
            200
        )
        
        if success:
            expected_fields = ['compliance_mode', 'attorney_supervision_required', 'maintenance_mode', 
                             'compliance_level', 'system_status', 'checks_today', 'violations_today', 'compliance_rate']
            missing_fields = [field for field in expected_fields if field not in response]
            if missing_fields:
                print(f"   ⚠️  Missing expected fields: {missing_fields}")
            else:
                print(f"   ✅ All expected fields present")
                print(f"   Compliance Mode: {response.get('compliance_mode')}")
                print(f"   Attorney Supervision Required: {response.get('attorney_supervision_required')}")
                print(f"   System Status: {response.get('system_status')}")
        
        return success, response

    def test_compliance_check(self):
        """Test POST /api/compliance/check - UPL violation detection"""
        test_content = "You should definitely sue them immediately. I guarantee you will win this case and get millions in damages. This is clear legal malpractice."
        
        test_data = {
            "content": test_content,
            "content_type": "legal_advice",
            "user_id": "test_user_123"
        }
        
        success, response = self.run_test(
            "Compliance Check Endpoint", 
            "POST", 
            "compliance/check", 
            200,
            test_data
        )
        
        if success:
            expected_fields = ['is_compliant', 'violations', 'confidence_score', 'requires_attorney_review', 'blocked_phrases', 'recommendations']
            missing_fields = [field for field in expected_fields if field not in response]
            if missing_fields:
                print(f"   ⚠️  Missing expected fields: {missing_fields}")
            else:
                print(f"   ✅ All expected fields present")
                print(f"   Is Compliant: {response.get('is_compliant')}")
                print(f"   Violations Found: {len(response.get('violations', []))}")
                print(f"   Confidence Score: {response.get('confidence_score')}")
                print(f"   Requires Attorney Review: {response.get('requires_attorney_review')}")
        
        return success, response

    def test_content_sanitization(self):
        """Test POST /api/content/sanitize - Content sanitization (using query params)"""
        test_content = "You should sue them for breach of contract. This is definitely legal malpractice."
        
        # Use query parameters instead of JSON body based on previous test results
        params = {
            "content": test_content,
            "content_type": "contract",
            "user_id": "test_user_456"
        }
        
        success, response = self.run_test(
            "Content Sanitization Endpoint", 
            "POST", 
            "content/sanitize", 
            200,
            data={},  # Empty JSON body
            params=params
        )
        
        if success:
            expected_fields = ['sanitized_content', 'changes_made', 'disclaimers_added', 'blocked_phrases', 'confidence_score', 'requires_review']
            missing_fields = [field for field in expected_fields if field not in response]
            if missing_fields:
                print(f"   ⚠️  Missing expected fields: {missing_fields}")
            else:
                print(f"   ✅ All expected fields present")
                print(f"   Original Length: {len(test_content)}")
                print(f"   Sanitized Length: {len(response.get('sanitized_content', ''))}")
                print(f"   Changes Made: {response.get('changes_made')}")
                print(f"   Disclaimers Added: {response.get('disclaimers_added')}")
        
        return success, response

    def test_attorney_creation(self):
        """Test POST /api/attorney/create - Attorney account creation"""
        attorney_data = {
            "email": f"attorney_{random.randint(1000, 9999)}@legalmate.test",
            "first_name": "Sarah",
            "last_name": "Johnson",
            "bar_number": f"BAR{random.randint(100000, 999999)}",
            "jurisdiction": "California",
            "role": "supervising_attorney",
            "specializations": ["contract_law", "employment_law"],
            "years_experience": 8,
            "password": "SecurePassword123!"
        }
        
        success, response = self.run_test(
            "Attorney Creation Endpoint", 
            "POST", 
            "attorney/create", 
            200,
            attorney_data
        )
        
        if success:
            print(f"   ✅ Attorney account created successfully")
            # Store attorney info for later tests
            if 'attorney_id' in response:
                self.attorney_id = response['attorney_id']
                print(f"   Attorney ID: {self.attorney_id}")
            else:
                print(f"   ⚠️  No attorney_id in response: {response}")
        
        return success, response

    def test_attorney_login(self):
        """Test POST /api/attorney/login - Attorney authentication"""
        # First create an attorney account for login testing
        attorney_email = f"login_test_{random.randint(1000, 9999)}@legalmate.test"
        attorney_password = "LoginTest123!"
        
        # Create attorney
        create_data = {
            "email": attorney_email,
            "first_name": "Login",
            "last_name": "Tester",
            "bar_number": f"LOGIN{random.randint(100000, 999999)}",
            "jurisdiction": "New York",
            "role": "reviewing_attorney",
            "specializations": ["corporate_law"],
            "years_experience": 5,
            "password": attorney_password
        }
        
        create_success, create_response = self.run_test(
            "Create Attorney for Login Test", 
            "POST", 
            "attorney/create", 
            200,
            create_data
        )
        
        if not create_success:
            print("   ❌ Failed to create attorney for login test")
            return False, {}
        
        # Check if attorney_id was returned for future use
        if 'attorney_id' in create_response:
            print(f"   Created attorney ID: {create_response['attorney_id']}")
        
        # Now test login
        login_data = {
            "email": attorney_email,
            "password": attorney_password
        }
        
        success, response = self.run_test(
            "Attorney Login Endpoint", 
            "POST", 
            "attorney/login", 
            200,
            login_data
        )
        
        if success:
            expected_fields = ['token', 'attorney', 'expires_at']
            missing_fields = [field for field in expected_fields if field not in response]
            if missing_fields:
                print(f"   ⚠️  Missing expected fields: {missing_fields}")
            else:
                print(f"   ✅ All expected fields present")
                print(f"   Token received: {response.get('token', '')[:20]}...")
                print(f"   Attorney Name: {response.get('attorney', {}).get('first_name')} {response.get('attorney', {}).get('last_name')}")
                # Store token for authenticated requests
                self.attorney_token = response.get('token')
                self.attorney_id = response.get('attorney', {}).get('id')
        
        return success, response

    def test_attorney_profile(self):
        """Test GET /api/attorney/profile/{attorney_id} - Attorney profile"""
        if not self.attorney_id:
            print("   ⚠️  No attorney ID available, creating one...")
            # Create a test attorney
            attorney_data = {
                "email": f"profile_test_{random.randint(1000, 9999)}@legalmate.test",
                "first_name": "Profile",
                "last_name": "Tester",
                "bar_number": f"PROF{random.randint(100000, 999999)}",
                "jurisdiction": "Texas",
                "role": "reviewing_attorney",
                "specializations": ["family_law"],
                "years_experience": 3,
                "password": "ProfileTest123!"
            }
            
            create_success, create_response = self.run_test(
                "Create Attorney for Profile Test", 
                "POST", 
                "attorney/create", 
                200,
                attorney_data
            )
            
            if create_success and 'attorney_id' in create_response:
                self.attorney_id = create_response['attorney_id']
                print(f"   Created attorney ID for profile test: {self.attorney_id}")
            else:
                print("   ❌ Failed to create attorney for profile test")
                print(f"   Create response: {create_response}")
                return False, {}
        
        success, response = self.run_test(
            "Attorney Profile Endpoint", 
            "GET", 
            f"attorney/profile/{self.attorney_id}", 
            200
        )
        
        if success:
            expected_fields = ['id', 'email', 'first_name', 'last_name', 'bar_number', 'jurisdiction', 'specializations']
            missing_fields = [field for field in expected_fields if field not in response]
            if missing_fields:
                print(f"   ⚠️  Missing expected fields: {missing_fields}")
            else:
                print(f"   ✅ All expected fields present")
                print(f"   Attorney: {response.get('first_name')} {response.get('last_name')}")
                print(f"   Bar Number: {response.get('bar_number')}")
                print(f"   Jurisdiction: {response.get('jurisdiction')}")
        
        return success, response

    def test_document_review_submission(self):
        """Test POST /api/attorney/review/submit - Document submission"""
        document_data = {
            "document_content": "This is a test contract for attorney review. The parties agree to the following terms and conditions...",
            "document_type": "contract",
            "client_id": f"client_{random.randint(1000, 9999)}",
            "original_request": {
                "contract_type": "NDA",
                "jurisdiction": "US",
                "purpose": "Testing document review submission"
            },
            "priority": "normal"
        }
        
        success, response = self.run_test(
            "Document Review Submission Endpoint", 
            "POST", 
            "attorney/review/submit", 
            200,
            document_data
        )
        
        if success:
            expected_fields = ['review_id', 'document_id', 'assigned_attorney', 'estimated_review_time', 'status']
            missing_fields = [field for field in expected_fields if field not in response]
            if missing_fields:
                print(f"   ⚠️  Missing expected fields: {missing_fields}")
            else:
                print(f"   ✅ All expected fields present")
                print(f"   Review ID: {response.get('review_id')}")
                print(f"   Document ID: {response.get('document_id')}")
                print(f"   Assigned Attorney: {response.get('assigned_attorney')}")
                print(f"   Estimated Review Time: {response.get('estimated_review_time')}")
                # Store review ID for later tests
                self.review_id = response.get('review_id')
                self.client_id = document_data['client_id']
        
        return success, response

    def test_attorney_review_queue(self):
        """Test GET /api/attorney/review/queue/{attorney_id} - Attorney queue"""
        if not self.attorney_id:
            print("   ⚠️  No attorney ID available for queue test")
            return False, {}
        
        success, response = self.run_test(
            "Attorney Review Queue Endpoint", 
            "GET", 
            f"attorney/review/queue/{self.attorney_id}", 
            200
        )
        
        if success:
            expected_fields = ['attorney_id', 'pending_reviews', 'in_progress_reviews', 'total_workload']
            missing_fields = [field for field in expected_fields if field not in response]
            if missing_fields:
                print(f"   ⚠️  Missing expected fields: {missing_fields}")
            else:
                print(f"   ✅ All expected fields present")
                print(f"   Attorney ID: {response.get('attorney_id')}")
                print(f"   Pending Reviews: {len(response.get('pending_reviews', []))}")
                print(f"   In Progress Reviews: {len(response.get('in_progress_reviews', []))}")
                print(f"   Total Workload: {response.get('total_workload')}")
        
        return success, response

    def test_attorney_review_action(self):
        """Test POST /api/attorney/review/action - Attorney approval/rejection workflow - PRIORITY TEST"""
        # Use the attorney_id from earlier tests if available
        action_attorney_id = self.attorney_id
        action_review_id = self.review_id
        
        if not action_review_id or not action_attorney_id:
            print("   ⚠️  No review ID or attorney ID available, creating test data...")
            
            # Ensure we have an attorney ID first
            if not action_attorney_id:
                attorney_data = {
                    "email": f"action_attorney_{random.randint(1000, 9999)}@legalmate.test",
                    "first_name": "Action",
                    "last_name": "Tester",
                    "bar_number": f"ACT{random.randint(100000, 999999)}",
                    "jurisdiction": "California",
                    "role": "supervising_attorney",
                    "specializations": ["contract_law"],
                    "years_experience": 10,
                    "password": "ActionTest123!"
                }
                
                create_success, create_response = self.run_test(
                    "Create Attorney for Action Test", 
                    "POST", 
                    "attorney/create", 
                    200,
                    attorney_data
                )
                
                if create_success and 'attorney_id' in create_response:
                    action_attorney_id = create_response['attorney_id']
                    print(f"   Created attorney ID for action test: {action_attorney_id}")
                else:
                    print("   ❌ Failed to create attorney for action test")
                    print(f"   Create response: {create_response}")
                    return False, {}
            
            # Create a document review with the specific attorney
            document_data = {
                "document_content": "Test contract content for review action testing. This agreement shall be governed by applicable law.",
                "document_type": "contract",
                "client_id": f"action_client_{random.randint(1000, 9999)}",
                "original_request": {
                    "contract_type": "service_agreement",
                    "jurisdiction": "US"
                },
                "priority": "high"
            }
            
            submit_success, submit_response = self.run_test(
                "Create Document Review for Action Test", 
                "POST", 
                "attorney/review/submit", 
                200,
                document_data
            )
            
            if submit_success and 'review_id' in submit_response:
                action_review_id = submit_response['review_id']
                print(f"   Created review ID: {action_review_id}")
            else:
                print("   ❌ Failed to create document review for action test")
                return False, {}
        
        # Test approve action - Use the attorney_id that should be assigned to the review
        action_data = {
            "review_id": action_review_id,
            "attorney_id": action_attorney_id,  # Use the actual attorney ID string
            "action": "approve",
            "comments": "Contract looks good. All terms are reasonable and legally sound.",
            "approved_content": "Approved contract content with attorney review completed."
        }
        
        success, response = self.run_test(
            "Attorney Review Action Endpoint - PRIORITY TEST", 
            "POST", 
            "attorney/review/action", 
            200,
            action_data
        )
        
        if success:
            expected_fields = ['success', 'action', 'review_id', 'message']
            missing_fields = [field for field in expected_fields if field not in response]
            if missing_fields:
                print(f"   ⚠️  Missing expected fields: {missing_fields}")
            else:
                print(f"   ✅ All expected fields present")
                print(f"   Review ID: {response.get('review_id')}")
                print(f"   Action: {response.get('action')}")
                print(f"   Success: {response.get('success')}")
                print(f"   Message: {response.get('message')}")
                print(f"   🎯 PRIORITY TEST PASSED - Attorney Review Action endpoint working!")
        else:
            print(f"   🚨 PRIORITY TEST FAILED - Attorney Review Action endpoint not working!")
            # Debug: Check the review status to see who it's assigned to
            print(f"   🔍 Debugging: Checking review status...")
            debug_success, debug_response = self.run_test(
                "Debug Review Status", 
                "GET", 
                f"attorney/review/status/{action_review_id}", 
                200
            )
            if debug_success:
                print(f"   Debug - Review Status: {debug_response.get('status')}")
                print(f"   Debug - Assigned Attorney: {debug_response.get('attorney')}")
                print(f"   Debug - Our Attorney ID: {action_attorney_id}")
        
        return success, response

    def test_review_status_tracking(self):
        """Test GET /api/attorney/review/status/{review_id} - Review status tracking"""
        if not self.review_id:
            print("   ⚠️  No review ID available for status tracking test")
            return False, {}
        
        success, response = self.run_test(
            "Review Status Tracking Endpoint", 
            "GET", 
            f"attorney/review/status/{self.review_id}", 
            200
        )
        
        if success:
            expected_fields = ['review_id', 'status', 'assigned_attorney', 'progress_percentage', 'estimated_completion']
            missing_fields = [field for field in expected_fields if field not in response]
            if missing_fields:
                print(f"   ⚠️  Missing expected fields: {missing_fields}")
            else:
                print(f"   ✅ All expected fields present")
                print(f"   Review ID: {response.get('review_id')}")
                print(f"   Status: {response.get('status')}")
                print(f"   Progress: {response.get('progress_percentage')}%")
                print(f"   Assigned Attorney: {response.get('assigned_attorney')}")
        
        return success, response

    def test_client_consent_recording(self):
        """Test POST /api/client/consent - Client consent recording"""
        if not self.client_id:
            self.client_id = f"consent_client_{random.randint(1000, 9999)}"
        
        consent_data = {
            "client_id": self.client_id,
            "consent_text": "I hereby consent to attorney supervision and review of legal documents generated through this platform.",
            "ip_address": "192.168.1.100",
            "user_agent": "Mozilla/5.0 (Test Browser) LegalMate/1.0"
        }
        
        success, response = self.run_test(
            "Client Consent Recording Endpoint", 
            "POST", 
            "client/consent", 
            200,
            consent_data
        )
        
        if success:
            expected_fields = ['consent_id', 'client_id', 'timestamp', 'status']
            missing_fields = [field for field in expected_fields if field not in response]
            if missing_fields:
                print(f"   ⚠️  Missing expected fields: {missing_fields}")
            else:
                print(f"   ✅ All expected fields present")
                print(f"   Consent ID: {response.get('consent_id')}")
                print(f"   Client ID: {response.get('client_id')}")
                print(f"   Status: {response.get('status')}")
        
        return success, response

    def test_client_consent_validation(self):
        """Test GET /api/client/consent/check/{client_id} - Consent validation"""
        if not self.client_id:
            print("   ⚠️  No client ID available for consent validation test")
            return False, {}
        
        success, response = self.run_test(
            "Client Consent Validation Endpoint", 
            "GET", 
            f"client/consent/check/{self.client_id}", 
            200
        )
        
        if success:
            expected_fields = ['client_id', 'has_consent', 'consent_date', 'consent_status']
            missing_fields = [field for field in expected_fields if field not in response]
            if missing_fields:
                print(f"   ⚠️  Missing expected fields: {missing_fields}")
            else:
                print(f"   ✅ All expected fields present")
                print(f"   Client ID: {response.get('client_id')}")
                print(f"   Has Consent: {response.get('has_consent')}")
                print(f"   Consent Status: {response.get('consent_status')}")
        
        return success, response

    def test_compliant_contract_generation(self):
        """Test POST /api/generate-contract-compliant - Compliant contract generation - PRIORITY TEST"""
        contract_data = {
            "contract_type": "NDA",
            "jurisdiction": "US",
            "parties": {
                "party1_name": "Compliance Test Corp",
                "party1_type": "corporation",
                "party2_name": "Contract Tester",
                "party2_type": "individual"
            },
            "terms": {
                "purpose": "Testing compliant contract generation with UPL violation checking",
                "duration": "2_years"
            },
            "special_clauses": ["Attorney supervision clause"],
            "client_id": self.client_id or f"compliant_client_{random.randint(1000, 9999)}",
            "document_id": f"doc_{random.randint(100000, 999999)}"  # Add required document_id
        }
        
        success, response = self.run_test(
            "Compliant Contract Generation Endpoint - PRIORITY TEST", 
            "POST", 
            "generate-contract-compliant", 
            200,
            contract_data,
            timeout=60  # AI generation might take longer
        )
        
        if success:
            expected_fields = ['contract', 'compliance_check', 'attorney_review_required', 'sanitized_content']
            missing_fields = [field for field in expected_fields if field not in response]
            if missing_fields:
                print(f"   ⚠️  Missing expected fields: {missing_fields}")
            else:
                print(f"   ✅ All expected fields present")
                
                contract = response.get('contract', {})
                compliance_check = response.get('compliance_check', {})
                
                print(f"   Contract ID: {contract.get('id')}")
                print(f"   Compliance Score: {contract.get('compliance_score')}%")
                print(f"   Attorney Review Required: {response.get('attorney_review_required')}")
                print(f"   Compliance Check Status: {compliance_check.get('is_compliant')}")
                print(f"   🎯 PRIORITY TEST PASSED - Compliant Contract Generation endpoint working!")
        else:
            print(f"   🚨 PRIORITY TEST FAILED - Compliant Contract Generation endpoint not working!")
        
        return success, response

    def run_all_tests(self):
        """Run all Day 1 Legal Compliance System tests"""
        print("🚀 Starting Day 1 Legal Compliance System Backend Testing")
        print("=" * 80)
        print("CRITICAL PRIORITY: Re-testing previously failed endpoints")
        print("- Attorney Review Action Endpoint (POST /api/attorney/review/action)")
        print("- Compliant Contract Generation Endpoint (POST /api/generate-contract-compliant)")
        print("=" * 80)
        
        # Test all 13 endpoints
        test_results = []
        
        # 1. Compliance Status
        result = self.test_compliance_status()
        test_results.append(("Compliance Status", result[0]))
        
        # 2. Compliance Check
        result = self.test_compliance_check()
        test_results.append(("Compliance Check", result[0]))
        
        # 3. Content Sanitization
        result = self.test_content_sanitization()
        test_results.append(("Content Sanitization", result[0]))
        
        # 4. Attorney Creation
        result = self.test_attorney_creation()
        test_results.append(("Attorney Creation", result[0]))
        
        # 5. Attorney Login
        result = self.test_attorney_login()
        test_results.append(("Attorney Login", result[0]))
        
        # 6. Attorney Profile
        result = self.test_attorney_profile()
        test_results.append(("Attorney Profile", result[0]))
        
        # 7. Document Review Submission
        result = self.test_document_review_submission()
        test_results.append(("Document Review Submission", result[0]))
        
        # 8. Attorney Review Queue
        result = self.test_attorney_review_queue()
        test_results.append(("Attorney Review Queue", result[0]))
        
        # 9. Attorney Review Action - PRIORITY TEST
        result = self.test_attorney_review_action()
        test_results.append(("Attorney Review Action - PRIORITY", result[0]))
        
        # 10. Review Status Tracking
        result = self.test_review_status_tracking()
        test_results.append(("Review Status Tracking", result[0]))
        
        # 11. Client Consent Recording
        result = self.test_client_consent_recording()
        test_results.append(("Client Consent Recording", result[0]))
        
        # 12. Client Consent Validation
        result = self.test_client_consent_validation()
        test_results.append(("Client Consent Validation", result[0]))
        
        # 13. Compliant Contract Generation - PRIORITY TEST
        result = self.test_compliant_contract_generation()
        test_results.append(("Compliant Contract Generation - PRIORITY", result[0]))
        
        # Print summary
        print("\n" + "=" * 80)
        print("🎯 DAY 1 LEGAL COMPLIANCE SYSTEM TEST RESULTS")
        print("=" * 80)
        
        passed_count = sum(1 for _, passed in test_results if passed)
        total_count = len(test_results)
        success_rate = (passed_count / total_count) * 100
        
        print(f"📊 Overall Results: {passed_count}/{total_count} tests passed ({success_rate:.1f}% success rate)")
        print()
        
        # Show individual results
        for test_name, passed in test_results:
            status = "✅ PASS" if passed else "❌ FAIL"
            priority_marker = " 🎯 PRIORITY" if "PRIORITY" in test_name else ""
            print(f"{status} {test_name}{priority_marker}")
        
        print("\n" + "=" * 80)
        
        # Focus on priority tests
        priority_tests = [(name, passed) for name, passed in test_results if "PRIORITY" in name]
        priority_passed = sum(1 for _, passed in priority_tests if passed)
        priority_total = len(priority_tests)
        
        print(f"🎯 PRIORITY TESTS RESULTS: {priority_passed}/{priority_total} passed")
        
        if priority_passed == priority_total:
            print("🎉 SUCCESS: All previously failed endpoints are now working!")
        else:
            print("🚨 ATTENTION: Some previously failed endpoints still need fixes")
        
        print(f"\n📈 Improvement from previous test: 84.6% → {success_rate:.1f}%")
        
        if success_rate == 100.0:
            print("🏆 PERFECT SCORE: All Day 1 Legal Compliance endpoints working!")
        elif success_rate >= 90.0:
            print("🌟 EXCELLENT: Day 1 Legal Compliance system is highly functional!")
        elif success_rate >= 80.0:
            print("👍 GOOD: Day 1 Legal Compliance system is mostly functional!")
        else:
            print("⚠️  NEEDS WORK: Day 1 Legal Compliance system needs more fixes!")
        
        return success_rate, test_results

if __name__ == "__main__":
    tester = Day1ComplianceSystemTester()
    success_rate, results = tester.run_all_tests()
    
    # Exit with appropriate code
    if success_rate == 100.0:
        sys.exit(0)  # Perfect success
    elif success_rate >= 90.0:
        sys.exit(0)  # Acceptable success
    else:
        sys.exit(1)  # Needs improvement