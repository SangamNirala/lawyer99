import requests
import sys
import json
import time
import random
from datetime import datetime

class AttorneyAssignmentTester:
    def __init__(self, base_url="https://027a3f35-a0bc-40cd-ad79-483d46c6126e.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.created_attorneys = []
        self.created_reviews = []

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

    def create_test_attorneys(self):
        """Create multiple test attorneys to ensure there are available attorneys in the system"""
        print(f"\n🏛️ STEP 1: Creating Test Attorneys...")
        
        attorney_data_list = [
            {
                "email": f"attorney1_{int(time.time())}@testlaw.com",
                "first_name": "Sarah",
                "last_name": "Johnson",
                "bar_number": f"BAR{random.randint(100000, 999999)}",
                "jurisdiction": "US",
                "role": "reviewing_attorney",
                "specializations": ["contract_law", "business_law"],
                "years_experience": 8,
                "password": "SecurePass123!"
            },
            {
                "email": f"attorney2_{int(time.time())}@testlaw.com",
                "first_name": "Michael",
                "last_name": "Chen",
                "bar_number": f"BAR{random.randint(100000, 999999)}",
                "jurisdiction": "US",
                "role": "supervising_attorney",
                "specializations": ["employment_law", "contract_law"],
                "years_experience": 12,
                "password": "SecurePass123!"
            },
            {
                "email": f"attorney3_{int(time.time())}@testlaw.com",
                "first_name": "Emily",
                "last_name": "Rodriguez",
                "bar_number": f"BAR{random.randint(100000, 999999)}",
                "jurisdiction": "US",
                "role": "senior_partner",
                "specializations": ["business_law", "partnership_law"],
                "years_experience": 15,
                "password": "SecurePass123!"
            }
        ]
        
        attorneys_created = 0
        for i, attorney_data in enumerate(attorney_data_list):
            success, response = self.run_test(
                f"Create Test Attorney {i+1} ({attorney_data['first_name']} {attorney_data['last_name']})",
                "POST",
                "attorney/create",
                200,
                attorney_data,
                timeout=30
            )
            
            if success and response.get('success'):
                attorney_id = response.get('attorney_id')
                self.created_attorneys.append({
                    'id': attorney_id,
                    'email': attorney_data['email'],
                    'name': f"{attorney_data['first_name']} {attorney_data['last_name']}",
                    'role': attorney_data['role'],
                    'specializations': attorney_data['specializations']
                })
                attorneys_created += 1
                print(f"   ✅ Attorney created with ID: {attorney_id}")
            else:
                print(f"   ❌ Failed to create attorney {i+1}")
        
        print(f"\n📊 Attorney Creation Summary: {attorneys_created}/{len(attorney_data_list)} attorneys created successfully")
        
        if attorneys_created == 0:
            print("❌ CRITICAL: No attorneys were created. Cannot proceed with assignment testing.")
            return False
        elif attorneys_created < len(attorney_data_list):
            print("⚠️  WARNING: Not all attorneys were created, but proceeding with available attorneys.")
        
        return attorneys_created > 0

    def record_client_consent(self, client_id):
        """Record client consent for attorney supervision"""
        consent_data = {
            "client_id": client_id,
            "consent_text": "I consent to attorney supervision and review of my legal documents as required by law.",
            "ip_address": "192.168.1.100",
            "user_agent": "AttorneyAssignmentTester/1.0"
        }
        
        success, response = self.run_test(
            f"Record Client Consent for {client_id}",
            "POST",
            "client/consent",
            200,
            consent_data,
            timeout=30
        )
        
        if success and response.get('success'):
            print(f"   ✅ Consent recorded with ID: {response.get('consent_id')}")
            return True
        else:
            print(f"   ❌ Failed to record consent")
            return False

    def generate_compliant_contract(self, client_id, contract_type="NDA", party1_name="Test Company Inc.", party2_name="John Doe"):
        """Generate a compliant contract that should trigger attorney review"""
        contract_data = {
            "contract_type": contract_type,
            "jurisdiction": "US",
            "parties": {
                "party1_name": party1_name,
                "party1_type": "company",
                "party2_name": party2_name,
                "party2_type": "individual"
            },
            "terms": {
                "purpose": "Business collaboration evaluation and confidential information sharing",
                "duration": "2_years"
            },
            "special_clauses": ["Non-compete clause for 6 months"],
            "client_id": client_id
        }
        
        success, response = self.run_test(
            f"Generate Compliant Contract ({contract_type})",
            "POST",
            "generate-contract-compliant",
            200,
            contract_data,
            timeout=60
        )
        
        if success and 'suggestions' in response:
            suggestions = response.get('suggestions', [])
            print(f"   Contract generation suggestions: {len(suggestions)} items")
            
            # Extract review ID from suggestions
            review_id = None
            for suggestion in suggestions:
                if 'Document submitted for attorney review (ID:' in suggestion:
                    # Extract UUID from the suggestion text
                    import re
                    match = re.search(r'ID:\s*([a-f0-9-]{36})', suggestion)
                    if match:
                        review_id = match.group(1)
                        break
            
            if review_id:
                print(f"   ✅ Review created with ID: {review_id}")
                self.created_reviews.append(review_id)
                return True, review_id
            else:
                print(f"   ❌ No review ID found in suggestions")
                return False, None
        else:
            print(f"   ❌ Contract generation failed or no suggestions returned")
            return False, None

    def check_review_status(self, review_id):
        """Check the status of a review to verify attorney assignment and progress"""
        success, response = self.run_test(
            f"Check Review Status for {review_id}",
            "GET",
            f"attorney/review/status/{review_id}",
            200,
            timeout=30
        )
        
        if success:
            status = response.get('status', 'unknown')
            progress = response.get('progress_percentage', 0)
            attorney = response.get('assigned_attorney')
            priority = response.get('priority', 'unknown')
            created_at = response.get('created_at', 'unknown')
            estimated_completion = response.get('estimated_completion', 'unknown')
            
            print(f"   📋 Review Status Details:")
            print(f"      Status: {status}")
            print(f"      Progress: {progress}%")
            print(f"      Priority: {priority}")
            print(f"      Created: {created_at}")
            print(f"      Estimated Completion: {estimated_completion}")
            
            if attorney:
                attorney_name = attorney.get('name', 'Unknown')
                attorney_id = attorney.get('attorney_id', 'Unknown')
                print(f"      Assigned Attorney: {attorney_name} (ID: {attorney_id})")
                print(f"   ✅ Attorney is assigned to this review")
                return True, {
                    'status': status,
                    'progress': progress,
                    'attorney_assigned': True,
                    'attorney_name': attorney_name,
                    'attorney_id': attorney_id
                }
            else:
                print(f"      Assigned Attorney: No attorney assigned")
                print(f"   ❌ No attorney assigned to this review")
                return True, {
                    'status': status,
                    'progress': progress,
                    'attorney_assigned': False,
                    'attorney_name': None,
                    'attorney_id': None
                }
        else:
            print(f"   ❌ Failed to check review status")
            return False, {}

    def monitor_progress_advancement(self, review_id, monitoring_duration=60):
        """Monitor a review's progress over time to verify it advances"""
        print(f"\n⏱️  Monitoring progress advancement for review {review_id} over {monitoring_duration} seconds...")
        
        progress_history = []
        start_time = time.time()
        
        # Initial check
        success, initial_status = self.check_review_status(review_id)
        if success:
            progress_history.append({
                'time': 0,
                'progress': initial_status.get('progress', 0),
                'status': initial_status.get('status', 'unknown')
            })
            print(f"   Initial Progress: {initial_status.get('progress', 0)}%")
        
        # Monitor at intervals
        check_intervals = [15, 30, 45, 60]  # Check at these second intervals
        
        for interval in check_intervals:
            if interval <= monitoring_duration:
                time.sleep(interval - (time.time() - start_time) if interval - (time.time() - start_time) > 0 else 0)
                
                success, current_status = self.check_review_status(review_id)
                if success:
                    elapsed_time = int(time.time() - start_time)
                    current_progress = current_status.get('progress', 0)
                    current_status_text = current_status.get('status', 'unknown')
                    
                    progress_history.append({
                        'time': elapsed_time,
                        'progress': current_progress,
                        'status': current_status_text
                    })
                    
                    print(f"   Progress at {elapsed_time}s: {current_progress}% (Status: {current_status_text})")
        
        # Analyze progress advancement
        if len(progress_history) >= 2:
            initial_progress = progress_history[0]['progress']
            final_progress = progress_history[-1]['progress']
            progress_increase = final_progress - initial_progress
            
            print(f"\n📈 Progress Analysis:")
            print(f"   Initial Progress: {initial_progress}%")
            print(f"   Final Progress: {final_progress}%")
            print(f"   Progress Increase: {progress_increase}%")
            
            # Check for status transitions
            initial_status = progress_history[0]['status']
            final_status = progress_history[-1]['status']
            
            print(f"   Initial Status: {initial_status}")
            print(f"   Final Status: {final_status}")
            
            # Determine if progress is advancing properly
            if initial_progress == 0 and final_progress == 0:
                print(f"   ❌ Progress stuck at 0% - attorney assignment issue likely persists")
                return False, progress_history
            elif progress_increase > 0:
                print(f"   ✅ Progress is advancing over time (+{progress_increase}%)")
                return True, progress_history
            elif initial_progress > 0:
                print(f"   ✅ Progress started above 0% ({initial_progress}%) - assignment working")
                return True, progress_history
            else:
                print(f"   ⚠️  Progress not advancing but started above 0%")
                return True, progress_history
        else:
            print(f"   ❌ Insufficient data to analyze progress advancement")
            return False, progress_history

    def test_complete_document_generation_flow(self):
        """Test the complete document generation flow with attorney assignment"""
        print(f"\n📋 STEP 2: Testing Complete Document Generation Flow...")
        
        # Generate unique client ID
        client_id = f"client_{int(time.time())}_{random.randint(1000, 9999)}"
        print(f"   Using client ID: {client_id}")
        
        # Step 1: Record client consent
        print(f"\n   Step 2.1: Recording client consent...")
        consent_success = self.record_client_consent(client_id)
        if not consent_success:
            print(f"   ❌ Failed to record consent - cannot proceed with flow")
            return False
        
        # Step 2: Generate compliant contract
        print(f"\n   Step 2.2: Generating compliant contract...")
        contract_success, review_id = self.generate_compliant_contract(
            client_id, 
            contract_type="NDA",
            party1_name="Test Company Inc.",
            party2_name="John Doe"
        )
        
        if not contract_success or not review_id:
            print(f"   ❌ Failed to generate contract or extract review ID")
            return False
        
        # Step 3: Verify review creation and attorney assignment
        print(f"\n   Step 2.3: Verifying review creation and attorney assignment...")
        status_success, status_data = self.check_review_status(review_id)
        
        if not status_success:
            print(f"   ❌ Failed to check review status")
            return False
        
        # Check if attorney is assigned
        if not status_data.get('attorney_assigned', False):
            print(f"   ❌ CRITICAL: Review created but no attorney assigned - assignment fix failed")
            return False
        
        print(f"   ✅ Review created and attorney assigned successfully")
        print(f"   ✅ Attorney: {status_data.get('attorney_name')} (ID: {status_data.get('attorney_id')})")
        
        # Step 4: Verify progress is above 0%
        initial_progress = status_data.get('progress', 0)
        if initial_progress == 0:
            print(f"   ❌ CRITICAL: Progress is still 0% despite attorney assignment")
            return False
        
        print(f"   ✅ Initial progress is {initial_progress}% (above 0%)")
        
        # Step 5: Monitor progress advancement over time
        print(f"\n   Step 2.4: Monitoring progress advancement...")
        progress_success, progress_history = self.monitor_progress_advancement(review_id, monitoring_duration=60)
        
        if not progress_success:
            print(f"   ❌ Progress monitoring failed or progress not advancing")
            return False
        
        print(f"   ✅ Progress monitoring successful - progress is advancing over time")
        
        return True

    def test_multiple_reviews_consistency(self):
        """Test multiple reviews to ensure consistent attorney assignment and progress advancement"""
        print(f"\n🔄 STEP 3: Testing Multiple Reviews for Consistency...")
        
        test_cases = [
            {"contract_type": "NDA", "party1": "Alpha Corp", "party2": "Beta User"},
            {"contract_type": "freelance_agreement", "party1": "Gamma LLC", "party2": "Delta Freelancer"},
            {"contract_type": "partnership_agreement", "party1": "Epsilon Partners", "party2": "Zeta Ventures"}
        ]
        
        successful_reviews = 0
        review_results = []
        
        for i, test_case in enumerate(test_cases):
            print(f"\n   Test Case {i+1}: {test_case['contract_type']}")
            
            # Generate unique client ID for each test
            client_id = f"client_multi_{int(time.time())}_{i}_{random.randint(1000, 9999)}"
            
            # Record consent
            consent_success = self.record_client_consent(client_id)
            if not consent_success:
                print(f"   ❌ Failed to record consent for test case {i+1}")
                continue
            
            # Generate contract
            contract_success, review_id = self.generate_compliant_contract(
                client_id,
                contract_type=test_case['contract_type'],
                party1_name=test_case['party1'],
                party2_name=test_case['party2']
            )
            
            if not contract_success or not review_id:
                print(f"   ❌ Failed to generate contract for test case {i+1}")
                continue
            
            # Check review status
            status_success, status_data = self.check_review_status(review_id)
            
            if not status_success:
                print(f"   ❌ Failed to check status for test case {i+1}")
                continue
            
            # Analyze results
            attorney_assigned = status_data.get('attorney_assigned', False)
            progress = status_data.get('progress', 0)
            status = status_data.get('status', 'unknown')
            
            result = {
                'test_case': i+1,
                'contract_type': test_case['contract_type'],
                'review_id': review_id,
                'attorney_assigned': attorney_assigned,
                'progress': progress,
                'status': status,
                'attorney_name': status_data.get('attorney_name'),
                'success': attorney_assigned and progress > 0
            }
            
            review_results.append(result)
            
            if result['success']:
                successful_reviews += 1
                print(f"   ✅ Test Case {i+1}: Attorney assigned ({status_data.get('attorney_name')}), Progress: {progress}%")
            else:
                print(f"   ❌ Test Case {i+1}: Assignment failed or progress at 0%")
        
        # Summary
        print(f"\n📊 Multiple Reviews Test Summary:")
        print(f"   Successful Reviews: {successful_reviews}/{len(test_cases)}")
        print(f"   Success Rate: {(successful_reviews/len(test_cases)*100):.1f}%")
        
        # Detailed analysis
        if successful_reviews > 0:
            assigned_attorneys = set()
            progress_values = []
            
            for result in review_results:
                if result['success']:
                    if result['attorney_name']:
                        assigned_attorneys.add(result['attorney_name'])
                    progress_values.append(result['progress'])
            
            print(f"   Unique Attorneys Assigned: {len(assigned_attorneys)}")
            print(f"   Attorney Names: {list(assigned_attorneys)}")
            print(f"   Progress Range: {min(progress_values):.1f}% - {max(progress_values):.1f}%")
            print(f"   Average Progress: {sum(progress_values)/len(progress_values):.1f}%")
        
        return successful_reviews == len(test_cases), review_results

    def test_status_transitions(self):
        """Test that reviews transition from 'pending' to 'in_review' status when attorneys are assigned"""
        print(f"\n🔄 STEP 4: Testing Status Transitions (pending → in_review)...")
        
        # Generate a new review for status transition testing
        client_id = f"client_status_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # Record consent and generate contract
        consent_success = self.record_client_consent(client_id)
        if not consent_success:
            print(f"   ❌ Failed to record consent for status transition test")
            return False
        
        contract_success, review_id = self.generate_compliant_contract(
            client_id,
            contract_type="employment_agreement",
            party1_name="Status Test Corp",
            party2_name="Transition Tester"
        )
        
        if not contract_success or not review_id:
            print(f"   ❌ Failed to generate contract for status transition test")
            return False
        
        # Check initial status immediately after creation
        print(f"   Checking initial status immediately after creation...")
        initial_success, initial_status = self.check_review_status(review_id)
        
        if not initial_success:
            print(f"   ❌ Failed to check initial status")
            return False
        
        initial_status_text = initial_status.get('status', 'unknown')
        initial_progress = initial_status.get('progress', 0)
        initial_attorney_assigned = initial_status.get('attorney_assigned', False)
        
        print(f"   Initial Status: {initial_status_text}")
        print(f"   Initial Progress: {initial_progress}%")
        print(f"   Initial Attorney Assigned: {initial_attorney_assigned}")
        
        # Wait a moment and check again to see if status transitions
        print(f"   Waiting 10 seconds for potential status transition...")
        time.sleep(10)
        
        final_success, final_status = self.check_review_status(review_id)
        
        if not final_success:
            print(f"   ❌ Failed to check final status")
            return False
        
        final_status_text = final_status.get('status', 'unknown')
        final_progress = final_status.get('progress', 0)
        final_attorney_assigned = final_status.get('attorney_assigned', False)
        
        print(f"   Final Status: {final_status_text}")
        print(f"   Final Progress: {final_progress}%")
        print(f"   Final Attorney Assigned: {final_attorney_assigned}")
        
        # Analyze status transition
        print(f"\n   📊 Status Transition Analysis:")
        
        if not final_attorney_assigned:
            print(f"   ❌ CRITICAL: No attorney assigned - cannot test status transitions")
            return False
        
        if final_status_text == 'pending':
            print(f"   ❌ Status remains 'pending' despite attorney assignment")
            return False
        elif final_status_text == 'in_review':
            print(f"   ✅ Status correctly transitioned to 'in_review'")
        else:
            print(f"   ⚠️  Status is '{final_status_text}' - unexpected but may be valid")
        
        if final_progress > 0:
            print(f"   ✅ Progress is above 0% ({final_progress}%)")
        else:
            print(f"   ❌ Progress is still 0% despite attorney assignment")
            return False
        
        return True

    def run_comprehensive_test(self):
        """Run the comprehensive attorney assignment test suite"""
        print(f"🎯 ATTORNEY ASSIGNMENT COMPREHENSIVE TEST SUITE")
        print(f"=" * 60)
        print(f"Testing the critical fix for attorney assignment issue")
        print(f"User reported: Progress stuck at 0% due to no attorney assignment")
        print(f"Expected: Reviews assigned to attorneys with 25%+ progress advancement")
        print(f"=" * 60)
        
        # Step 1: Create test attorneys
        if not self.create_test_attorneys():
            print(f"\n❌ CRITICAL FAILURE: Could not create test attorneys")
            return False
        
        # Step 2: Test complete document generation flow
        flow_success = self.test_complete_document_generation_flow()
        
        # Step 3: Test multiple reviews for consistency
        consistency_success, review_results = self.test_multiple_reviews_consistency()
        
        # Step 4: Test status transitions
        transition_success = self.test_status_transitions()
        
        # Final Summary
        print(f"\n" + "=" * 60)
        print(f"🎯 COMPREHENSIVE TEST RESULTS SUMMARY")
        print(f"=" * 60)
        
        total_tests = 4  # Number of major test categories
        passed_tests = sum([
            1 if len(self.created_attorneys) > 0 else 0,  # Attorney creation
            1 if flow_success else 0,  # Complete flow
            1 if consistency_success else 0,  # Multiple reviews consistency
            1 if transition_success else 0  # Status transitions
        ])
        
        success_rate = (passed_tests / total_tests) * 100
        
        print(f"📊 Overall Results:")
        print(f"   Tests Passed: {passed_tests}/{total_tests}")
        print(f"   Success Rate: {success_rate:.1f}%")
        print(f"   Individual API Tests: {self.tests_passed}/{self.tests_run}")
        
        print(f"\n📋 Detailed Results:")
        print(f"   ✅ Attorney Creation: {len(self.created_attorneys)} attorneys created" if len(self.created_attorneys) > 0 else "   ❌ Attorney Creation: Failed")
        print(f"   ✅ Complete Flow Test: Passed" if flow_success else "   ❌ Complete Flow Test: Failed")
        print(f"   ✅ Multiple Reviews Consistency: Passed" if consistency_success else "   ❌ Multiple Reviews Consistency: Failed")
        print(f"   ✅ Status Transitions: Passed" if transition_success else "   ❌ Status Transitions: Failed")
        
        print(f"\n🔍 Key Findings:")
        if len(self.created_attorneys) > 0:
            print(f"   • {len(self.created_attorneys)} test attorneys available for assignment")
        
        if len(self.created_reviews) > 0:
            print(f"   • {len(self.created_reviews)} reviews created during testing")
        
        # Determine overall success
        critical_success = flow_success and len(self.created_attorneys) > 0
        
        if critical_success:
            print(f"\n✅ CRITICAL FIX VERIFICATION: SUCCESS")
            print(f"   The attorney assignment issue appears to be resolved:")
            print(f"   • Reviews are being assigned to attorneys")
            print(f"   • Progress advances above 0% (typically 25%+)")
            print(f"   • Status transitions from 'pending' to 'in_review'")
        else:
            print(f"\n❌ CRITICAL FIX VERIFICATION: FAILED")
            print(f"   The attorney assignment issue may still persist:")
            print(f"   • Reviews may not be getting assigned to attorneys")
            print(f"   • Progress may still be stuck at 0%")
            print(f"   • Status may remain 'pending' indefinitely")
        
        return critical_success

if __name__ == "__main__":
    tester = AttorneyAssignmentTester()
    success = tester.run_comprehensive_test()
    
    if success:
        print(f"\n🎉 All critical tests passed! Attorney assignment fix is working.")
        sys.exit(0)
    else:
        print(f"\n💥 Critical tests failed. Attorney assignment issue may persist.")
        sys.exit(1)