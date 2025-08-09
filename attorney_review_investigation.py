import requests
import sys
import json
import time
from datetime import datetime

class AttorneyReviewInvestigator:
    def __init__(self, base_url="https://8cd68d5c-4981-470b-b1c0-9982d2b4a8d2.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.specific_review_id = "b5f7f23-24c1-4780-878b-afb6cf3814f"
        self.attorney_id = None

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

    def test_specific_review_status(self):
        """Test the specific review ID mentioned in the issue"""
        print(f"\n🎯 INVESTIGATING SPECIFIC REVIEW ID: {self.specific_review_id}")
        
        success, response = self.run_test(
            f"Review Status for ID {self.specific_review_id}",
            "GET",
            f"attorney/review/status/{self.specific_review_id}",
            200
        )
        
        if success and response:
            print(f"   📋 REVIEW DETAILS:")
            print(f"      Review ID: {response.get('review_id', 'N/A')}")
            print(f"      Status: {response.get('status', 'N/A')}")
            print(f"      Created At: {response.get('created_at', 'N/A')}")
            print(f"      Estimated Completion: {response.get('estimated_completion', 'N/A')}")
            print(f"      Progress Percentage: {response.get('progress_percentage', 'N/A')}%")
            print(f"      Priority: {response.get('priority', 'N/A')}")
            
            # Check attorney assignment
            attorney_info = response.get('attorney', {})
            if attorney_info:
                print(f"      Assigned Attorney: {attorney_info.get('name', 'N/A')} (ID: {attorney_info.get('id', 'N/A')})")
                print(f"      Attorney Role: {attorney_info.get('role', 'N/A')}")
                self.attorney_id = attorney_info.get('id')
            else:
                print(f"      ⚠️  NO ATTORNEY ASSIGNED - This could be the issue!")
            
            # Check if review is overdue
            status = response.get('status', '').lower()
            created_at = response.get('created_at', '')
            if status == 'pending' and created_at:
                print(f"      🚨 REVIEW STATUS: {status.upper()}")
                try:
                    created_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    current_time = datetime.now(created_time.tzinfo)
                    time_diff = current_time - created_time
                    hours_elapsed = time_diff.total_seconds() / 3600
                    print(f"      ⏰ Time Elapsed: {hours_elapsed:.1f} hours")
                    
                    if hours_elapsed > 24:
                        print(f"      🚨 OVERDUE: Review has been pending for {hours_elapsed:.1f} hours (>24h)")
                    elif hours_elapsed > 4:
                        print(f"      ⚠️  DELAYED: Review has been pending for {hours_elapsed:.1f} hours (>4h expected)")
                    else:
                        print(f"      ✅ WITHIN TIMEFRAME: Review pending for {hours_elapsed:.1f} hours")
                except Exception as e:
                    print(f"      ⚠️  Could not calculate time elapsed: {e}")
            
            return True, response
        else:
            print(f"   ❌ CRITICAL: Could not retrieve review status for {self.specific_review_id}")
            return False, {}

    def test_attorney_availability(self):
        """Check if there are any attorneys available in the system"""
        print(f"\n🔍 CHECKING ATTORNEY AVAILABILITY")
        
        # First, try to create a test attorney to ensure the system has attorneys
        test_attorney_data = {
            "email": "test.attorney@legalmate.ai",
            "first_name": "Test",
            "last_name": "Attorney",
            "bar_number": "TEST123456",
            "jurisdiction": "US",
            "role": "reviewing_attorney",
            "specializations": ["contract_law", "business_law"],
            "years_experience": 5,
            "password": "TestPassword123!"
        }
        
        success, response = self.run_test(
            "Create Test Attorney",
            "POST",
            "attorney/create",
            200,
            test_attorney_data
        )
        
        if success:
            print(f"   ✅ Test attorney created successfully")
            # Try to get the attorney ID from response or use a test ID
            if 'attorney_id' in response:
                test_attorney_id = response['attorney_id']
            else:
                # Use a test attorney ID - we'll check if it exists
                test_attorney_id = "test-attorney-001"
            
            # Now check attorney profile
            profile_success, profile_response = self.run_test(
                f"Get Attorney Profile",
                "GET",
                f"attorney/profile/{test_attorney_id}",
                200
            )
            
            if profile_success and profile_response:
                print(f"   📋 ATTORNEY PROFILE:")
                print(f"      Name: {profile_response.get('first_name', '')} {profile_response.get('last_name', '')}")
                print(f"      Email: {profile_response.get('email', 'N/A')}")
                print(f"      Role: {profile_response.get('role', 'N/A')}")
                print(f"      Available: {profile_response.get('available', 'N/A')}")
                print(f"      Current Reviews: {profile_response.get('current_review_count', 'N/A')}")
                print(f"      Specializations: {profile_response.get('specializations', [])}")
                print(f"      Last Login: {profile_response.get('last_login', 'N/A')}")
                
                self.attorney_id = test_attorney_id
                return True, profile_response
            else:
                print(f"   ⚠️  Could not retrieve attorney profile")
        else:
            print(f"   ⚠️  Could not create test attorney - checking existing attorneys")
            
        # If creation failed, try some common attorney IDs
        test_attorney_ids = [
            "attorney-001",
            "test-attorney-001", 
            "default-attorney",
            "reviewing-attorney-001"
        ]
        
        for attorney_id in test_attorney_ids:
            profile_success, profile_response = self.run_test(
                f"Check Attorney {attorney_id}",
                "GET",
                f"attorney/profile/{attorney_id}",
                200
            )
            
            if profile_success:
                print(f"   ✅ Found existing attorney: {attorney_id}")
                self.attorney_id = attorney_id
                return True, profile_response
        
        print(f"   ❌ CRITICAL: No attorneys found in the system!")
        return False, {}

    def test_attorney_queue(self):
        """Test attorney review queue to see if reviews are being assigned properly"""
        if not self.attorney_id:
            print(f"\n⚠️  Skipping attorney queue test - no attorney ID available")
            return True, {}
        
        print(f"\n🔍 CHECKING ATTORNEY REVIEW QUEUE")
        
        success, response = self.run_test(
            f"Attorney Review Queue for {self.attorney_id}",
            "GET",
            f"attorney/review/queue/{self.attorney_id}",
            200
        )
        
        if success and response:
            print(f"   📋 ATTORNEY QUEUE STATUS:")
            print(f"      Queue Length: {response.get('queue_length', 'N/A')}")
            print(f"      Attorney ID: {response.get('attorney_id', 'N/A')}")
            
            reviews = response.get('reviews', [])
            if reviews:
                print(f"      📄 REVIEWS IN QUEUE ({len(reviews)}):")
                for i, review in enumerate(reviews[:5]):  # Show first 5 reviews
                    print(f"         {i+1}. Review ID: {review.get('review_id', 'N/A')}")
                    print(f"            Status: {review.get('status', 'N/A')}")
                    print(f"            Priority: {review.get('priority', 'N/A')}")
                    print(f"            Document Type: {review.get('document_type', 'N/A')}")
                    print(f"            Created: {review.get('created_at', 'N/A')}")
                    
                    # Check if our specific review is in this queue
                    if review.get('review_id') == self.specific_review_id:
                        print(f"            🎯 FOUND OUR REVIEW IN QUEUE!")
                        
                if len(reviews) > 5:
                    print(f"         ... and {len(reviews) - 5} more reviews")
            else:
                print(f"      ⚠️  NO REVIEWS IN QUEUE - This could explain why the review is stuck!")
            
            return True, response
        else:
            print(f"   ❌ Could not retrieve attorney queue")
            return False, {}

    def test_attorney_assignment_logic(self):
        """Test the attorney assignment logic by submitting a test document"""
        print(f"\n🔍 TESTING ATTORNEY ASSIGNMENT LOGIC")
        
        test_document_data = {
            "document_content": "This is a test contract for attorney assignment testing. It contains standard business terms and conditions for a service agreement between two parties.",
            "document_type": "contract",
            "client_id": "test-client-assignment-001",
            "original_request": {
                "contract_type": "service_agreement",
                "parties": {
                    "party1_name": "Test Company",
                    "party2_name": "Test Client"
                }
            },
            "priority": "normal"
        }
        
        success, response = self.run_test(
            "Submit Document for Attorney Review",
            "POST",
            "attorney/review/submit",
            200,
            test_document_data
        )
        
        if success and response:
            print(f"   📋 DOCUMENT SUBMISSION RESULT:")
            print(f"      Review ID: {response.get('review_id', 'N/A')}")
            print(f"      Status: {response.get('status', 'N/A')}")
            print(f"      Assigned Attorney: {response.get('assigned_attorney_id', 'N/A')}")
            print(f"      Estimated Review Time: {response.get('estimated_review_time', 'N/A')}")
            print(f"      Priority: {response.get('priority', 'N/A')}")
            
            # Check if attorney was assigned
            assigned_attorney = response.get('assigned_attorney_id')
            if assigned_attorney:
                print(f"      ✅ ATTORNEY SUCCESSFULLY ASSIGNED: {assigned_attorney}")
                
                # Now check the status of this new review
                new_review_id = response.get('review_id')
                if new_review_id:
                    time.sleep(2)  # Wait a moment for the system to process
                    
                    status_success, status_response = self.run_test(
                        f"Check New Review Status",
                        "GET",
                        f"attorney/review/status/{new_review_id}",
                        200
                    )
                    
                    if status_success:
                        print(f"      📋 NEW REVIEW STATUS:")
                        print(f"         Status: {status_response.get('status', 'N/A')}")
                        print(f"         Progress: {status_response.get('progress_percentage', 'N/A')}%")
                        print(f"         Attorney: {status_response.get('attorney', {}).get('name', 'N/A')}")
                        
                        if status_response.get('status') == 'pending':
                            print(f"         ✅ Review properly created and assigned")
                        else:
                            print(f"         ⚠️  Unexpected status: {status_response.get('status')}")
            else:
                print(f"      ❌ CRITICAL: NO ATTORNEY ASSIGNED - Assignment logic may be broken!")
            
            return True, response
        else:
            print(f"   ❌ Document submission failed - assignment logic cannot be tested")
            return False, {}

    def test_review_progression(self):
        """Test if reviews can progress beyond 0%"""
        if not self.attorney_id:
            print(f"\n⚠️  Skipping review progression test - no attorney ID available")
            return True, {}
        
        print(f"\n🔍 TESTING REVIEW PROGRESSION")
        
        # Try to perform an action on the specific review if it exists
        review_action_data = {
            "review_id": self.specific_review_id,
            "attorney_id": self.attorney_id,
            "action": "approve",
            "comments": "Test approval to check if review can progress",
            "approved_content": "Approved test content"
        }
        
        success, response = self.run_test(
            "Test Review Action (Approve)",
            "POST",
            "attorney/review/action",
            200,
            review_action_data
        )
        
        if success:
            print(f"   ✅ Review action successful")
            print(f"      Action Result: {response.get('message', 'N/A')}")
            print(f"      New Status: {response.get('new_status', 'N/A')}")
            
            # Check the review status again to see if it progressed
            time.sleep(2)
            status_success, status_response = self.run_test(
                f"Check Review Status After Action",
                "GET",
                f"attorney/review/status/{self.specific_review_id}",
                200
            )
            
            if status_success:
                print(f"   📋 UPDATED REVIEW STATUS:")
                print(f"      Status: {status_response.get('status', 'N/A')}")
                print(f"      Progress: {status_response.get('progress_percentage', 'N/A')}%")
                
                if status_response.get('progress_percentage', 0) > 0:
                    print(f"      ✅ REVIEW PROGRESSED BEYOND 0%!")
                else:
                    print(f"      ❌ REVIEW STILL AT 0% - Progression issue confirmed")
            
            return True, response
        else:
            print(f"   ⚠️  Review action failed - may indicate authorization or assignment issues")
            
            # Try with a different action
            review_action_data['action'] = 'request_revision'
            review_action_data['revision_requests'] = [
                {"section": "terms", "comment": "Please clarify payment terms"}
            ]
            
            success2, response2 = self.run_test(
                "Test Review Action (Request Revision)",
                "POST",
                "attorney/review/action",
                200,
                review_action_data
            )
            
            if success2:
                print(f"   ✅ Revision request successful")
                return True, response2
            else:
                print(f"   ❌ All review actions failed - system may have authorization issues")
                return False, {}

    def investigate_overdue_status(self):
        """Investigate why the review is showing as 'Overdue'"""
        print(f"\n🔍 INVESTIGATING OVERDUE STATUS")
        
        # Check compliance system status
        compliance_success, compliance_response = self.run_test(
            "Check Compliance System Status",
            "GET",
            "compliance/status",
            200
        )
        
        if compliance_success:
            print(f"   📋 COMPLIANCE SYSTEM STATUS:")
            print(f"      Compliance Mode: {compliance_response.get('compliance_mode', 'N/A')}")
            print(f"      Attorney Supervision Required: {compliance_response.get('attorney_supervision_required', 'N/A')}")
            print(f"      System Status: {compliance_response.get('system_status', 'N/A')}")
            print(f"      Checks Today: {compliance_response.get('checks_today', 'N/A')}")
            print(f"      Violations Today: {compliance_response.get('violations_today', 'N/A')}")
            print(f"      Compliance Rate: {compliance_response.get('compliance_rate', 'N/A')}%")
            
            if compliance_response.get('system_status') != 'operational':
                print(f"      🚨 COMPLIANCE SYSTEM NOT OPERATIONAL - This could cause review delays!")
        
        # Check if there are any system-wide issues
        print(f"\n   🔍 Checking for system-wide attorney supervision issues...")
        
        # Try to check multiple review statuses to see if it's a widespread issue
        test_review_ids = [
            self.specific_review_id,
            "test-review-001",
            "test-review-002"
        ]
        
        overdue_count = 0
        total_checked = 0
        
        for review_id in test_review_ids:
            status_success, status_response = self.run_test(
                f"Check Review {review_id}",
                "GET",
                f"attorney/review/status/{review_id}",
                200
            )
            
            if status_success:
                total_checked += 1
                status = status_response.get('status', '').lower()
                created_at = status_response.get('created_at', '')
                
                if status == 'pending' and created_at:
                    try:
                        created_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                        current_time = datetime.now(created_time.tzinfo)
                        time_diff = current_time - created_time
                        hours_elapsed = time_diff.total_seconds() / 3600
                        
                        if hours_elapsed > 24:
                            overdue_count += 1
                            print(f"      🚨 Review {review_id}: OVERDUE ({hours_elapsed:.1f}h)")
                        elif hours_elapsed > 4:
                            print(f"      ⚠️  Review {review_id}: DELAYED ({hours_elapsed:.1f}h)")
                        else:
                            print(f"      ✅ Review {review_id}: ON TIME ({hours_elapsed:.1f}h)")
                    except:
                        pass
        
        if total_checked > 0:
            overdue_rate = (overdue_count / total_checked) * 100
            print(f"\n   📊 OVERDUE ANALYSIS:")
            print(f"      Reviews Checked: {total_checked}")
            print(f"      Overdue Reviews: {overdue_count}")
            print(f"      Overdue Rate: {overdue_rate:.1f}%")
            
            if overdue_rate > 50:
                print(f"      🚨 HIGH OVERDUE RATE - Systemic issue likely!")
            elif overdue_rate > 0:
                print(f"      ⚠️  Some overdue reviews - May indicate capacity issues")
            else:
                print(f"      ✅ No widespread overdue issues detected")

    def run_investigation(self):
        """Run the complete attorney review system investigation"""
        print("=" * 80)
        print("🔍 ATTORNEY REVIEW SYSTEM INVESTIGATION")
        print("=" * 80)
        print(f"Target Review ID: {self.specific_review_id}")
        print(f"API Base URL: {self.api_url}")
        print("=" * 80)
        
        # Run all investigation tests
        self.test_specific_review_status()
        self.test_attorney_availability()
        self.test_attorney_queue()
        self.test_attorney_assignment_logic()
        self.test_review_progression()
        self.investigate_overdue_status()
        
        # Summary
        print("\n" + "=" * 80)
        print("📊 INVESTIGATION SUMMARY")
        print("=" * 80)
        print(f"Tests Run: {self.tests_run}")
        print(f"Tests Passed: {self.tests_passed}")
        print(f"Success Rate: {(self.tests_passed/self.tests_run*100):.1f}%" if self.tests_run > 0 else "0%")
        
        if self.tests_passed < self.tests_run:
            print(f"\n🚨 ISSUES DETECTED:")
            print(f"   - {self.tests_run - self.tests_passed} tests failed")
            print(f"   - Review system may have critical issues")
            print(f"   - Immediate attention required")
        else:
            print(f"\n✅ ALL TESTS PASSED:")
            print(f"   - Attorney review system appears functional")
            print(f"   - Issue may be specific to the mentioned review ID")
        
        print("=" * 80)

if __name__ == "__main__":
    investigator = AttorneyReviewInvestigator()
    investigator.run_investigation()