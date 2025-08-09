#!/usr/bin/env python3
"""
Progress Percentage Monitoring Test
===================================

This test specifically addresses the user-reported issue where progress percentage
stays stuck at 0% after document generation. It tests the complete flow:
1. Client consent recording
2. Compliant contract generation  
3. Review status monitoring over time
4. Attorney assignment verification
5. Progress calculation logic

The test monitors progress percentage over multiple calls to identify if it's
actually advancing from 0% as intended.
"""

import requests
import sys
import json
import time
import uuid
from datetime import datetime

class ProgressMonitoringTester:
    def __init__(self, base_url="https://b2e17d7f-d715-45b1-acd0-e51dd70d439b.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.review_id = None
        self.client_id = None
        
    def log(self, message, level="INFO"):
        """Log messages with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
        
    def run_test(self, name, method, endpoint, expected_status, data=None, timeout=30):
        """Run a single API test with detailed logging"""
        url = f"{self.api_url}/{endpoint}" if not endpoint.startswith('http') else endpoint
        headers = {'Content-Type': 'application/json'}

        self.tests_run += 1
        self.log(f"🔍 Testing {name}")
        self.log(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=timeout)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=timeout)
            elif method == 'PUT':
                response = requests.put(url, json=data, headers=headers, timeout=timeout)

            self.log(f"   Status: {response.status_code}")
            
            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                self.log(f"✅ PASSED - Status: {response.status_code}")
                try:
                    response_data = response.json()
                    if isinstance(response_data, dict):
                        self.log(f"   Response keys: {list(response_data.keys())}")
                    return True, response_data
                except:
                    return True, response.text
            else:
                self.log(f"❌ FAILED - Expected {expected_status}, got {response.status_code}", "ERROR")
                try:
                    error_data = response.json()
                    self.log(f"   Error: {error_data}", "ERROR")
                    return False, error_data
                except:
                    self.log(f"   Error text: {response.text}", "ERROR")
                    return False, response.text
                    
        except Exception as e:
            self.log(f"❌ EXCEPTION - {str(e)}", "ERROR")
            return False, str(e)

    def check_existing_stuck_reviews(self):
        """Check if there are existing reviews stuck at 0% progress"""
        self.log("🔍 STEP 1: Checking for existing stuck reviews in database")
        
        # Try to get some existing review IDs by creating a test attorney first
        success, attorney_data = self.run_test(
            "Create Test Attorney for Review Check",
            "POST", 
            "attorney/create",
            200,
            {
                "email": f"test_attorney_{int(time.time())}@example.com",
                "first_name": "Test",
                "last_name": "Attorney",
                "bar_number": f"BAR{int(time.time())}",
                "jurisdiction": "US",
                "role": "reviewing_attorney",
                "specializations": ["contract_law"],
                "years_experience": 5,
                "password": "TestPassword123!"
            }
        )
        
        if success:
            attorney_id = attorney_data.get('attorney_id')
            self.log(f"✅ Test attorney created: {attorney_id}")
            
            # Check attorney queue to see if there are any existing reviews
            success, queue_data = self.run_test(
                "Check Attorney Review Queue",
                "GET",
                f"attorney/review/queue/{attorney_id}",
                200
            )
            
            if success and queue_data.get('reviews'):
                self.log(f"📋 Found {len(queue_data['reviews'])} existing reviews in queue")
                for review in queue_data['reviews']:
                    review_id = review.get('review_id')
                    progress = review.get('progress_percentage', 0)
                    status = review.get('status', 'unknown')
                    self.log(f"   Review {review_id}: {progress}% progress, status: {status}")
                    
                    if progress == 0 and status == 'pending':
                        self.log(f"🚨 FOUND STUCK REVIEW: {review_id} at 0% progress", "WARNING")
            else:
                self.log("📋 No existing reviews found in attorney queue")
        else:
            self.log("❌ Could not create test attorney to check existing reviews", "ERROR")

    def test_document_generation_flow(self):
        """Test the complete document generation flow with progress monitoring"""
        self.log("🔍 STEP 2: Testing complete document generation flow")
        
        # Generate unique client ID
        timestamp = int(time.time())
        random_suffix = str(uuid.uuid4())[:8]
        self.client_id = f"client_{timestamp}_{random_suffix}"
        self.log(f"📝 Using client ID: {self.client_id}")
        
        # Step 2.1: Record client consent
        self.log("📝 Step 2.1: Recording client consent")
        consent_success, consent_data = self.run_test(
            "Record Client Consent",
            "POST",
            "client/consent",
            200,
            {
                "client_id": self.client_id,
                "consent_text": "I consent to attorney supervision and legal document review",
                "ip_address": "192.168.1.100",
                "user_agent": "Progress Monitoring Test"
            }
        )
        
        if not consent_success:
            self.log("❌ Failed to record consent - cannot proceed with flow", "ERROR")
            return False
            
        self.log(f"✅ Consent recorded: {consent_data.get('consent_id')}")
        
        # Step 2.2: Verify consent status
        self.log("📝 Step 2.2: Verifying consent status")
        consent_check_success, consent_check_data = self.run_test(
            "Check Client Consent Status",
            "GET",
            f"client/consent/check/{self.client_id}",
            200
        )
        
        if consent_check_success:
            has_consent = consent_check_data.get('has_consent', False)
            self.log(f"✅ Consent status verified: {has_consent}")
        
        # Step 2.3: Generate compliant contract
        self.log("📝 Step 2.3: Generating compliant contract")
        contract_data = {
            "contract_type": "NDA",
            "parties": {
                "party1_name": "Test Company Inc.",
                "party1_type": "company",
                "party2_name": "John Doe",
                "party2_type": "individual"
            },
            "terms": {
                "purpose": "Business collaboration evaluation",
                "duration": "2 years",
                "confidentiality_level": "high"
            },
            "jurisdiction": "US",
            "client_id": self.client_id
        }
        
        contract_success, contract_response = self.run_test(
            "Generate Compliant Contract",
            "POST",
            "generate-contract-compliant",
            200,
            contract_data
        )
        
        if not contract_success:
            self.log("❌ Failed to generate contract - cannot proceed with progress monitoring", "ERROR")
            return False
            
        # Extract review ID from suggestions
        suggestions = contract_response.get('suggestions', [])
        self.review_id = None
        
        for suggestion in suggestions:
            if 'Document submitted for attorney review' in suggestion and 'ID:' in suggestion:
                # Extract UUID from suggestion text
                import re
                uuid_pattern = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
                match = re.search(uuid_pattern, suggestion)
                if match:
                    self.review_id = match.group()
                    break
        
        if not self.review_id:
            self.log("❌ Could not extract review ID from contract generation response", "ERROR")
            self.log(f"   Suggestions: {suggestions}")
            return False
            
        self.log(f"✅ Contract generated successfully, Review ID: {self.review_id}")
        return True

    def monitor_progress_over_time(self):
        """Monitor the progress percentage over multiple calls to see if it advances"""
        if not self.review_id:
            self.log("❌ No review ID available for progress monitoring", "ERROR")
            return False
            
        self.log("🔍 STEP 3: Monitoring progress percentage over time")
        self.log(f"📊 Monitoring review ID: {self.review_id}")
        
        progress_history = []
        monitoring_duration = 60  # Monitor for 60 seconds
        check_interval = 10  # Check every 10 seconds
        checks = monitoring_duration // check_interval
        
        for i in range(checks):
            self.log(f"📊 Progress check {i+1}/{checks}")
            
            success, status_data = self.run_test(
                f"Review Status Check #{i+1}",
                "GET",
                f"attorney/review/status/{self.review_id}",
                200
            )
            
            if success:
                progress = status_data.get('progress_percentage', 0)
                status = status_data.get('status', 'unknown')
                attorney = status_data.get('assigned_attorney')
                estimated_completion = status_data.get('estimated_completion')
                
                progress_entry = {
                    'check_number': i+1,
                    'timestamp': datetime.now().isoformat(),
                    'progress_percentage': progress,
                    'status': status,
                    'has_attorney': attorney is not None,
                    'attorney_id': attorney.get('attorney_id') if attorney else None,
                    'estimated_completion': estimated_completion
                }
                progress_history.append(progress_entry)
                
                self.log(f"   Progress: {progress}%, Status: {status}")
                self.log(f"   Attorney assigned: {'Yes' if attorney else 'No'}")
                if attorney:
                    self.log(f"   Attorney ID: {attorney.get('attorney_id')}")
                    self.log(f"   Attorney name: {attorney.get('first_name')} {attorney.get('last_name')}")
                self.log(f"   Estimated completion: {estimated_completion}")
                
            else:
                self.log(f"❌ Failed to get status on check {i+1}", "ERROR")
                progress_history.append({
                    'check_number': i+1,
                    'timestamp': datetime.now().isoformat(),
                    'error': 'Failed to retrieve status'
                })
            
            if i < checks - 1:  # Don't sleep after the last check
                self.log(f"⏳ Waiting {check_interval} seconds before next check...")
                time.sleep(check_interval)
        
        # Analyze progress history
        self.analyze_progress_history(progress_history)
        return progress_history

    def analyze_progress_history(self, progress_history):
        """Analyze the progress history to identify issues"""
        self.log("🔍 STEP 4: Analyzing progress history")
        
        if not progress_history:
            self.log("❌ No progress history to analyze", "ERROR")
            return
        
        # Extract progress values
        progress_values = []
        statuses = []
        attorney_assignments = []
        
        for entry in progress_history:
            if 'error' not in entry:
                progress_values.append(entry['progress_percentage'])
                statuses.append(entry['status'])
                attorney_assignments.append(entry['has_attorney'])
        
        if not progress_values:
            self.log("❌ No valid progress data collected", "ERROR")
            return
        
        # Analysis
        self.log("📊 PROGRESS ANALYSIS RESULTS:")
        self.log(f"   Total checks: {len(progress_history)}")
        self.log(f"   Valid responses: {len(progress_values)}")
        self.log(f"   Progress values: {progress_values}")
        self.log(f"   Status values: {statuses}")
        self.log(f"   Attorney assigned: {attorney_assignments}")
        
        # Check if progress is stuck at 0%
        if all(p == 0 for p in progress_values):
            self.log("🚨 CRITICAL ISSUE: Progress is stuck at 0% across all checks", "ERROR")
            self.log("   This confirms the user-reported issue!")
        elif all(p == progress_values[0] for p in progress_values):
            self.log(f"🚨 ISSUE: Progress is stuck at {progress_values[0]}% (not advancing)", "WARNING")
        else:
            self.log("✅ Progress is advancing over time")
            min_progress = min(progress_values)
            max_progress = max(progress_values)
            self.log(f"   Progress range: {min_progress}% to {max_progress}%")
        
        # Check attorney assignment
        if not any(attorney_assignments):
            self.log("🚨 CRITICAL ISSUE: No attorney assigned throughout monitoring period", "ERROR")
            self.log("   This is likely the root cause of stuck progress!")
        elif all(attorney_assignments):
            self.log("✅ Attorney consistently assigned")
        else:
            self.log("⚠️  Attorney assignment inconsistent during monitoring")
        
        # Check status progression
        unique_statuses = list(set(statuses))
        self.log(f"   Status progression: {' → '.join(statuses) if len(set(statuses)) > 1 else statuses[0] + ' (no change)'}")
        
        if len(unique_statuses) == 1 and unique_statuses[0] == 'pending':
            self.log("🚨 CRITICAL ISSUE: Review stuck in 'pending' status", "ERROR")
            self.log("   Reviews should progress from 'pending' to 'in_review' when attorney is assigned")

    def test_attorney_assignment_system(self):
        """Test the attorney assignment system specifically"""
        self.log("🔍 STEP 5: Testing attorney assignment system")
        
        # Create multiple attorneys to test assignment
        attorneys_created = []
        for i in range(3):
            success, attorney_data = self.run_test(
                f"Create Test Attorney #{i+1}",
                "POST",
                "attorney/create",
                200,
                {
                    "email": f"test_attorney_{int(time.time())}_{i}@example.com",
                    "first_name": f"Attorney{i+1}",
                    "last_name": "Test",
                    "bar_number": f"BAR{int(time.time())}{i}",
                    "jurisdiction": "US",
                    "role": "reviewing_attorney",
                    "specializations": ["contract_law", "business_law"],
                    "years_experience": 5 + i,
                    "password": "TestPassword123!"
                }
            )
            
            if success:
                attorneys_created.append(attorney_data.get('attorney_id'))
                self.log(f"✅ Attorney #{i+1} created: {attorney_data.get('attorney_id')}")
        
        self.log(f"📊 Created {len(attorneys_created)} test attorneys")
        
        # Test cleanup stuck reviews endpoint
        self.log("🧹 Testing cleanup stuck reviews functionality")
        cleanup_success, cleanup_data = self.run_test(
            "Cleanup Stuck Reviews",
            "POST",
            "attorney/review/cleanup-stuck",
            200
        )
        
        if cleanup_success:
            fixed_count = cleanup_data.get('fixed_count', 0)
            self.log(f"✅ Cleanup completed: {fixed_count} reviews fixed")
            if fixed_count > 0:
                self.log("   Some stuck reviews were found and fixed!")
        
        return len(attorneys_created) > 0

    def run_comprehensive_test(self):
        """Run the complete comprehensive test suite"""
        self.log("🚀 STARTING COMPREHENSIVE PROGRESS MONITORING TEST")
        self.log("=" * 60)
        
        start_time = time.time()
        
        try:
            # Step 1: Check existing stuck reviews
            self.check_existing_stuck_reviews()
            
            # Step 2: Test document generation flow
            flow_success = self.test_document_generation_flow()
            
            if flow_success:
                # Step 3: Monitor progress over time
                progress_history = self.monitor_progress_over_time()
                
                # Step 4: Test attorney assignment system
                self.test_attorney_assignment_system()
            else:
                self.log("❌ Document generation flow failed - skipping progress monitoring", "ERROR")
            
        except Exception as e:
            self.log(f"❌ Test suite failed with exception: {str(e)}", "ERROR")
        
        # Final summary
        end_time = time.time()
        duration = end_time - start_time
        
        self.log("=" * 60)
        self.log("🏁 TEST SUITE COMPLETED")
        self.log(f"📊 Tests run: {self.tests_run}")
        self.log(f"✅ Tests passed: {self.tests_passed}")
        self.log(f"❌ Tests failed: {self.tests_run - self.tests_passed}")
        self.log(f"📈 Success rate: {(self.tests_passed/self.tests_run*100):.1f}%")
        self.log(f"⏱️  Total duration: {duration:.1f} seconds")
        
        if self.review_id:
            self.log(f"🔍 Review ID for further investigation: {self.review_id}")
        if self.client_id:
            self.log(f"👤 Client ID used: {self.client_id}")

def main():
    """Main test execution"""
    print("🔍 Progress Percentage Monitoring Test")
    print("=====================================")
    print("Testing the user-reported issue where progress stays at 0%")
    print()
    
    tester = ProgressMonitoringTester()
    tester.run_comprehensive_test()

if __name__ == "__main__":
    main()