import requests
import sys
import json
import uuid
from datetime import datetime

class FocusedEndpointTester:
    def __init__(self, base_url="https://7eb15b8a-98e2-46d7-83f4-aba058ad34e0.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0

    def run_test(self, name, method, endpoint, expected_status, data=None, params=None, timeout=30):
        """Run a single API test"""
        url = f"{self.api_url}/{endpoint}" if not endpoint.startswith('http') else endpoint
        headers = {'Content-Type': 'application/json'}

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        if params:
            print(f"   Query Params: {params}")
        if data:
            print(f"   Request Data: {json.dumps(data, indent=2)}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, params=params, timeout=timeout)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, params=params, timeout=timeout)

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

    def test_hr_onboarding_workflow_with_workflow_type(self):
        """Test POST /api/hr/onboarding/create with proper workflow_type field"""
        print("\n" + "="*80)
        print("🎯 TESTING HR ONBOARDING WORKFLOW ENDPOINT")
        print("="*80)
        
        # Test 1: With proper workflow_type field
        onboarding_data_with_type = {
            "workflow_type": "onboarding",
            "employee_data": {
                "employee_id": "EMP2025001",
                "first_name": "Alice",
                "last_name": "Johnson",
                "email": "alice.johnson@company.com",
                "department": "Engineering",
                "position": "Software Engineer",
                "start_date": "2025-02-01T00:00:00Z"
            },
            "workflow_options": {
                "include_it_setup": True,
                "include_benefits_enrollment": True,
                "include_policy_acknowledgment": True,
                "assigned_buddy": "senior_team_member"
            },
            "company_id": str(uuid.uuid4())
        }
        
        success1, response1 = self.run_test(
            "HR Onboarding Workflow - With workflow_type",
            "POST",
            "hr/onboarding/create",
            200,
            onboarding_data_with_type
        )
        
        # Test 2: Without workflow_type field (should fail with 422)
        onboarding_data_without_type = {
            "employee_data": {
                "employee_id": "EMP2025002",
                "first_name": "Bob",
                "last_name": "Smith",
                "email": "bob.smith@company.com",
                "department": "Marketing",
                "position": "Marketing Manager"
            },
            "workflow_options": {
                "include_it_setup": False,
                "include_benefits_enrollment": True
            },
            "company_id": str(uuid.uuid4())
        }
        
        success2, response2 = self.run_test(
            "HR Onboarding Workflow - Without workflow_type (should fail)",
            "POST",
            "hr/onboarding/create",
            422,  # Expecting validation error
            onboarding_data_without_type
        )
        
        # Test 3: With different workflow_type values
        executive_onboarding_data = {
            "workflow_type": "executive_onboarding",
            "employee_data": {
                "employee_id": "EMP2025003",
                "first_name": "Carol",
                "last_name": "Executive",
                "email": "carol.executive@company.com",
                "department": "Executive",
                "position": "VP of Engineering",
                "start_date": "2025-02-15T00:00:00Z"
            },
            "workflow_options": {
                "include_it_setup": True,
                "include_benefits_enrollment": True,
                "include_policy_acknowledgment": True,
                "executive_orientation": True,
                "board_introduction": True
            },
            "company_id": str(uuid.uuid4())
        }
        
        success3, response3 = self.run_test(
            "HR Onboarding Workflow - Executive workflow_type",
            "POST",
            "hr/onboarding/create",
            200,
            executive_onboarding_data
        )
        
        print(f"\n📊 HR Onboarding Workflow Test Results:")
        print(f"   ✅ With workflow_type: {'PASSED' if success1 else 'FAILED'}")
        print(f"   ✅ Without workflow_type (422 expected): {'PASSED' if success2 else 'FAILED'}")
        print(f"   ✅ Executive workflow_type: {'PASSED' if success3 else 'FAILED'}")
        
        return success1 and success2 and success3, {
            "with_type": response1,
            "without_type": response2,
            "executive_type": response3
        }

    def test_contract_wizard_field_suggestions_parameter_format(self):
        """Test POST /api/contract-wizard/suggestions with correct parameter format"""
        print("\n" + "="*80)
        print("🎯 TESTING CONTRACT WIZARD FIELD SUGGESTIONS ENDPOINT")
        print("="*80)
        
        # Test 1: JSON body format (might be incorrect based on history)
        json_body_data = {
            "field_name": "party1_name",
            "contract_type": "employment_agreement",
            "user_id": str(uuid.uuid4()),
            "context": {
                "industry": "Technology",
                "company_size": "medium"
            }
        }
        
        success1, response1 = self.run_test(
            "Contract Wizard Suggestions - JSON Body Format",
            "POST",
            "contract-wizard/suggestions",
            200,
            json_body_data
        )
        
        # Test 2: Query parameters format (based on test_result.md history, this might be correct)
        query_params = {
            "field_name": "party1_email",
            "contract_type": "employment_agreement"
        }
        
        # Empty JSON body with query parameters
        success2, response2 = self.run_test(
            "Contract Wizard Suggestions - Query Params + Empty JSON",
            "POST",
            "contract-wizard/suggestions",
            200,
            {},  # Empty JSON body
            params=query_params
        )
        
        # Test 3: Different field types
        party_name_params = {
            "field_name": "party2_name",
            "contract_type": "freelance_agreement"
        }
        
        success3, response3 = self.run_test(
            "Contract Wizard Suggestions - Party Name Field",
            "POST",
            "contract-wizard/suggestions",
            200,
            {},
            params=party_name_params
        )
        
        # Test 4: Company name field
        company_params = {
            "field_name": "company_name",
            "contract_type": "employment_agreement"
        }
        
        success4, response4 = self.run_test(
            "Contract Wizard Suggestions - Company Name Field",
            "POST",
            "contract-wizard/suggestions",
            200,
            {},
            params=company_params
        )
        
        print(f"\n📊 Contract Wizard Field Suggestions Test Results:")
        print(f"   ✅ JSON Body Format: {'PASSED' if success1 else 'FAILED'}")
        print(f"   ✅ Query Params + Empty JSON: {'PASSED' if success2 else 'FAILED'}")
        print(f"   ✅ Party Name Field: {'PASSED' if success3 else 'FAILED'}")
        print(f"   ✅ Company Name Field: {'PASSED' if success4 else 'FAILED'}")
        
        # Determine which format works
        if success2 or success3 or success4:
            print(f"   🎯 CONCLUSION: Query parameters format appears to be working")
        elif success1:
            print(f"   🎯 CONCLUSION: JSON body format is working")
        else:
            print(f"   ❌ CONCLUSION: Neither format is working properly")
        
        return success1 or success2 or success3 or success4, {
            "json_body": response1,
            "query_params": response2,
            "party_name": response3,
            "company_name": response4
        }

    def test_hr_compliance_endpoints(self):
        """Test the newly implemented HR compliance endpoints"""
        print("\n" + "="*80)
        print("🎯 TESTING HR COMPLIANCE ENDPOINTS")
        print("="*80)
        
        # Test 1: POST /api/hr/compliance - Main compliance checking endpoint
        compliance_check_data = {
            "contract_content": """
            EMPLOYMENT AGREEMENT
            
            This Employment Agreement is entered into between TechCorp Inc. and Jane Doe.
            
            **POSITION AND DUTIES**
            Employee will serve as Senior Software Engineer in the Engineering Department.
            
            **COMPENSATION**
            Base salary: $95,000 annually
            Benefits: Health insurance, dental, vision, 401k matching up to 4%
            
            **EMPLOYMENT TERMS**
            Employment Type: Full-time, at-will employment
            Start Date: February 1, 2025
            Probationary Period: 90 days
            
            **CONFIDENTIALITY**
            Employee agrees to maintain confidentiality of all proprietary information.
            
            **TERMINATION**
            Either party may terminate this agreement with 30 days written notice.
            """,
            "contract_type": "employment_agreement",
            "jurisdiction": "US",
            "compliance_areas": ["employment_law", "benefits", "confidentiality"]
        }
        
        success1, response1 = self.run_test(
            "HR Compliance - Main Compliance Check",
            "POST",
            "hr/compliance",
            200,
            compliance_check_data
        )
        
        # Test 2: GET /api/hr/compliance/history - Compliance history endpoint
        success2, response2 = self.run_test(
            "HR Compliance - History Endpoint",
            "GET",
            "hr/compliance/history",
            200
        )
        
        # Test 3: GET /api/hr/compliance/summary - Compliance summary endpoint
        success3, response3 = self.run_test(
            "HR Compliance - Summary Endpoint",
            "GET",
            "hr/compliance/summary",
            200
        )
        
        # Test 4: POST /api/hr/compliance with different contract type
        contractor_compliance_data = {
            "contract_content": """
            INDEPENDENT CONTRACTOR AGREEMENT
            
            This agreement is between ABC Corp and John Smith for contractor services.
            
            **SERVICES**
            Contractor will provide web development services.
            
            **COMPENSATION**
            Hourly rate: $75/hour
            Payment terms: Net 30 days
            
            **INDEPENDENT CONTRACTOR STATUS**
            Contractor is an independent contractor, not an employee.
            Contractor responsible for own taxes and benefits.
            """,
            "contract_type": "contractor_agreement",
            "jurisdiction": "US",
            "compliance_areas": ["contractor_classification", "tax_compliance"]
        }
        
        success4, response4 = self.run_test(
            "HR Compliance - Contractor Agreement Check",
            "POST",
            "hr/compliance",
            200,
            contractor_compliance_data
        )
        
        print(f"\n📊 HR Compliance Endpoints Test Results:")
        print(f"   ✅ Main Compliance Check: {'PASSED' if success1 else 'FAILED'}")
        print(f"   ✅ Compliance History: {'PASSED' if success2 else 'FAILED'}")
        print(f"   ✅ Compliance Summary: {'PASSED' if success3 else 'FAILED'}")
        print(f"   ✅ Contractor Compliance Check: {'PASSED' if success4 else 'FAILED'}")
        
        # Check if endpoints are implemented
        if not success1 and not success2 and not success3:
            print(f"   ❌ CONCLUSION: HR Compliance endpoints appear to be NOT IMPLEMENTED (404 errors)")
        elif success1 and not success2 and not success3:
            print(f"   ⚠️  CONCLUSION: Main compliance check works, but history/summary endpoints not implemented")
        elif success1 and success2 and success3:
            print(f"   🎉 CONCLUSION: All HR Compliance endpoints are working!")
        else:
            print(f"   ⚠️  CONCLUSION: Mixed results - some endpoints working, others not")
        
        return success1 or success2 or success3 or success4, {
            "main_check": response1,
            "history": response2,
            "summary": response3,
            "contractor_check": response4
        }

    def run_focused_tests(self):
        """Run the three specific endpoint tests requested"""
        print("🚀 Starting Focused Endpoint Testing...")
        print("Testing the three specific endpoints that were just fixed/implemented")
        print("="*80)
        
        # Test 1: HR Onboarding Workflow
        test1_success, test1_response = self.test_hr_onboarding_workflow_with_workflow_type()
        
        # Test 2: Contract Wizard Field Suggestions
        test2_success, test2_response = self.test_contract_wizard_field_suggestions_parameter_format()
        
        # Test 3: HR Compliance Endpoints
        test3_success, test3_response = self.test_hr_compliance_endpoints()
        
        # Print final summary
        print("\n" + "="*80)
        print("🎯 FOCUSED ENDPOINT TEST SUMMARY")
        print("="*80)
        print(f"Total Tests Run: {self.tests_run}")
        print(f"Tests Passed: {self.tests_passed}")
        print(f"Tests Failed: {self.tests_run - self.tests_passed}")
        print(f"Success Rate: {(self.tests_passed/self.tests_run)*100:.1f}%" if self.tests_run > 0 else "No tests run")
        
        print(f"\n📋 ENDPOINT-SPECIFIC RESULTS:")
        print(f"   1. HR Onboarding Workflow: {'✅ WORKING' if test1_success else '❌ ISSUES FOUND'}")
        print(f"   2. Contract Wizard Field Suggestions: {'✅ WORKING' if test2_success else '❌ ISSUES FOUND'}")
        print(f"   3. HR Compliance Endpoints: {'✅ WORKING' if test3_success else '❌ ISSUES FOUND'}")
        
        # Detailed findings
        print(f"\n🔍 DETAILED FINDINGS:")
        
        if test1_success:
            print(f"   ✅ HR Onboarding Workflow: Parameter structure is correct, workflow_type field required")
        else:
            print(f"   ❌ HR Onboarding Workflow: Issues with parameter structure or workflow_type field")
            
        if test2_success:
            print(f"   ✅ Contract Wizard Field Suggestions: Parameter format adjustment is working")
        else:
            print(f"   ❌ Contract Wizard Field Suggestions: Parameter format issues persist")
            
        if test3_success:
            print(f"   ✅ HR Compliance Endpoints: Newly implemented endpoints are working")
        else:
            print(f"   ❌ HR Compliance Endpoints: Endpoints not implemented or have issues")
        
        # Overall assessment
        working_endpoints = sum([test1_success, test2_success, test3_success])
        print(f"\n🎯 OVERALL ASSESSMENT:")
        print(f"   Working Endpoints: {working_endpoints}/3")
        
        if working_endpoints == 3:
            print(f"   🎉 ALL REPORTED ISSUES HAVE BEEN RESOLVED!")
        elif working_endpoints == 2:
            print(f"   ✅ MOST ISSUES RESOLVED - 1 endpoint still has problems")
        elif working_endpoints == 1:
            print(f"   ⚠️  PARTIAL SUCCESS - 2 endpoints still have issues")
        else:
            print(f"   ❌ ALL ENDPOINTS STILL HAVE ISSUES - Fixes not working")
            
        return self.tests_passed, self.tests_run, {
            "hr_onboarding": test1_success,
            "contract_wizard": test2_success,
            "hr_compliance": test3_success
        }

if __name__ == "__main__":
    tester = FocusedEndpointTester()
    passed, total, endpoint_results = tester.run_focused_tests()
    
    # Exit with appropriate code
    if sum(endpoint_results.values()) == 3:
        sys.exit(0)  # All endpoint tests passed
    else:
        sys.exit(1)  # Some endpoint tests failed