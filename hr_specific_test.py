import requests
import sys
import json
import uuid
from datetime import datetime, timedelta

class HRSpecificEndpointTester:
    def __init__(self, base_url="https://b2e17d7f-d715-45b1-acd0-e51dd70d439b.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.detailed_errors = []

    def run_test(self, name, method, endpoint, expected_status, data=None, timeout=30):
        """Run a single API test with detailed error reporting"""
        url = f"{self.api_url}/{endpoint}" if not endpoint.startswith('http') else endpoint
        headers = {'Content-Type': 'application/json'}

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        print(f"   Method: {method}")
        if data:
            print(f"   Request Data: {json.dumps(data, indent=2)}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=timeout)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=timeout)
            elif method == 'PUT':
                response = requests.put(url, json=data, headers=headers, timeout=timeout)

            print(f"   Status: {response.status_code}")
            
            # Always capture detailed response information
            try:
                response_data = response.json()
                print(f"   Response: {json.dumps(response_data, indent=2)}")
            except:
                response_data = response.text
                print(f"   Response Text: {response_data}")
            
            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                return True, response_data
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                
                # Capture detailed error information
                error_detail = {
                    "test_name": name,
                    "expected_status": expected_status,
                    "actual_status": response.status_code,
                    "url": url,
                    "method": method,
                    "request_data": data,
                    "response_data": response_data,
                    "error_analysis": self.analyze_error(response.status_code, response_data)
                }
                self.detailed_errors.append(error_detail)
                
                return False, response_data

        except requests.exceptions.Timeout:
            error_msg = f"Request timed out after {timeout} seconds"
            print(f"❌ Failed - {error_msg}")
            self.detailed_errors.append({
                "test_name": name,
                "error_type": "timeout",
                "error_message": error_msg,
                "url": url,
                "method": method,
                "request_data": data
            })
            return False, {}
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"❌ Failed - {error_msg}")
            self.detailed_errors.append({
                "test_name": name,
                "error_type": "exception",
                "error_message": error_msg,
                "url": url,
                "method": method,
                "request_data": data
            })
            return False, {}

    def analyze_error(self, status_code, response_data):
        """Analyze error responses to provide insights"""
        analysis = []
        
        if status_code == 404:
            analysis.append("Endpoint not found - may not be implemented yet")
        elif status_code == 422:
            analysis.append("Validation error - request data format issue")
            if isinstance(response_data, dict):
                if 'detail' in response_data:
                    analysis.append(f"Validation details: {response_data['detail']}")
        elif status_code == 500:
            analysis.append("Internal server error - backend implementation issue")
        elif status_code == 400:
            analysis.append("Bad request - parameter structure or format issue")
            
        return analysis

    def test_hr_onboarding_workflow_create(self):
        """Test POST /api/hr/onboarding/create - HR Onboarding Workflow with workflow_type field"""
        print("\n" + "="*80)
        print("🎯 TESTING HR ONBOARDING WORKFLOW ENDPOINT")
        print("="*80)
        
        # Test 1: Standard request format (what might be expected)
        standard_request = {
            "workflow_type": "onboarding",
            "employee_data": {
                "employee_id": "EMP001",
                "first_name": "Sarah",
                "last_name": "Johnson",
                "email": "sarah.johnson@company.com",
                "department": "Engineering",
                "position": "Senior Developer",
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
            "HR Onboarding Workflow - Standard Format with workflow_type",
            "POST",
            "hr/onboarding/create",
            200,
            standard_request
        )
        
        # Test 2: Alternative format without workflow_type (to see what happens)
        alternative_request = {
            "employee_data": {
                "employee_id": "EMP002",
                "first_name": "Michael",
                "last_name": "Chen",
                "email": "michael.chen@company.com",
                "department": "Product",
                "position": "Product Manager",
                "start_date": "2025-02-15T00:00:00Z"
            },
            "workflow_options": {
                "template": "standard",
                "include_it_setup": True,
                "include_benefits_enrollment": True
            },
            "company_id": str(uuid.uuid4())
        }
        
        success2, response2 = self.run_test(
            "HR Onboarding Workflow - Without workflow_type field",
            "POST",
            "hr/onboarding/create",
            200,
            alternative_request
        )
        
        # Test 3: Different workflow types
        executive_request = {
            "workflow_type": "executive_onboarding",
            "employee_data": {
                "employee_id": "EMP003",
                "first_name": "Jennifer",
                "last_name": "Williams",
                "email": "jennifer.williams@company.com",
                "department": "Executive",
                "position": "VP of Engineering",
                "start_date": "2025-03-01T00:00:00Z"
            },
            "workflow_options": {
                "include_executive_briefing": True,
                "include_board_introduction": True,
                "include_equity_setup": True
            },
            "company_id": str(uuid.uuid4())
        }
        
        success3, response3 = self.run_test(
            "HR Onboarding Workflow - Executive workflow_type",
            "POST",
            "hr/onboarding/create",
            200,
            executive_request
        )
        
        return success1 or success2 or success3, {
            "standard": response1,
            "alternative": response2,
            "executive": response3
        }

    def test_contract_wizard_field_suggestions(self):
        """Test POST /api/contract-wizard/suggestions - Contract Wizard Field Suggestions parameter format"""
        print("\n" + "="*80)
        print("🎯 TESTING CONTRACT WIZARD FIELD SUGGESTIONS ENDPOINT")
        print("="*80)
        
        # Test 1: Standard format (based on existing code structure)
        standard_request = {
            "field_name": "employment_type",
            "contract_type": "employment_agreement",
            "context": {
                "position": "Software Engineer",
                "department": "Engineering",
                "company_size": "medium",
                "industry": "Technology"
            }
        }
        
        success1, response1 = self.run_test(
            "Contract Wizard Suggestions - Standard Format",
            "POST",
            "contract-wizard/suggestions",
            200,
            standard_request
        )
        
        # Test 2: Alternative format with user/company IDs
        alternative_request = {
            "field_name": "salary",
            "user_id": str(uuid.uuid4()),
            "company_id": str(uuid.uuid4()),
            "contract_type": "employment_agreement",
            "current_step": 2,
            "partial_data": {
                "position": "Senior Developer",
                "department": "Engineering",
                "location": "San Francisco, CA",
                "experience_level": "senior"
            }
        }
        
        success2, response2 = self.run_test(
            "Contract Wizard Suggestions - With User/Company IDs",
            "POST",
            "contract-wizard/suggestions",
            200,
            alternative_request
        )
        
        # Test 3: HR-specific field suggestions
        hr_request = {
            "field_name": "benefits_package",
            "contract_type": "offer_letter",
            "context": {
                "position": "HR Manager",
                "department": "Human Resources",
                "employment_type": "full_time",
                "company_size": "large",
                "industry": "Technology"
            }
        }
        
        success3, response3 = self.run_test(
            "Contract Wizard Suggestions - HR Benefits Field",
            "POST",
            "contract-wizard/suggestions",
            200,
            hr_request
        )
        
        # Test 4: Minimal format to test required fields
        minimal_request = {
            "field_name": "party1_name"
        }
        
        success4, response4 = self.run_test(
            "Contract Wizard Suggestions - Minimal Format",
            "POST",
            "contract-wizard/suggestions",
            200,
            minimal_request
        )
        
        return success1 or success2 or success3 or success4, {
            "standard": response1,
            "alternative": response2,
            "hr_specific": response3,
            "minimal": response4
        }

    def test_hr_compliance_endpoint(self):
        """Test GET/POST /api/hr/compliance endpoint - Check if it exists and test functionality"""
        print("\n" + "="*80)
        print("🎯 TESTING HR COMPLIANCE ENDPOINT")
        print("="*80)
        
        # Test 1: Check if GET endpoint exists
        success1, response1 = self.run_test(
            "HR Compliance - GET endpoint check",
            "GET",
            "hr/compliance",
            200
        )
        
        # Test 2: Check POST endpoint with compliance data
        compliance_request = {
            "company_id": str(uuid.uuid4()),
            "compliance_area": "employment_law",
            "requirement": "Equal Employment Opportunity Compliance",
            "description": "Ensure all hiring practices comply with EEO regulations",
            "jurisdiction": "US",
            "due_date": "2025-03-31T00:00:00Z",
            "status": "pending",
            "risk_level": "high",
            "documents_required": [
                "EEO-1 Report",
                "Hiring Process Documentation",
                "Training Records"
            ],
            "assigned_to": "HR Manager",
            "notes": "Annual compliance review required"
        }
        
        success2, response2 = self.run_test(
            "HR Compliance - POST create compliance item",
            "POST",
            "hr/compliance",
            200,
            compliance_request
        )
        
        # Test 3: Alternative compliance check format
        compliance_check_request = {
            "contract_content": """
            EMPLOYMENT AGREEMENT
            
            This Employment Agreement is entered into between TechCorp Inc. and Jane Doe.
            
            Position: Senior Software Engineer
            Department: Engineering
            Salary: $120,000 annually
            Benefits: Health insurance, dental, vision, 401k matching
            Employment Type: Full-time, at-will employment
            Start Date: February 1, 2025
            
            The employee agrees to comply with all company policies and procedures.
            """,
            "contract_type": "employment_agreement",
            "jurisdictions": ["US", "CA"],
            "compliance_areas": ["employment_law", "benefits", "tax"]
        }
        
        success3, response3 = self.run_test(
            "HR Compliance - Contract compliance check",
            "POST",
            "hr/compliance/check",
            200,
            compliance_check_request
        )
        
        # Test 4: Check if compliance endpoint exists under different path
        success4, response4 = self.run_test(
            "HR Compliance - Alternative path check",
            "GET",
            "hr/compliance/requirements",
            200
        )
        
        # Test 5: Try the general compliance-check endpoint for HR content
        general_compliance_request = {
            "contract_content": compliance_check_request["contract_content"],
            "jurisdictions": ["US"]
        }
        
        success5, response5 = self.run_test(
            "HR Compliance - Via general compliance-check endpoint",
            "POST",
            "compliance-check",
            200,
            general_compliance_request
        )
        
        return success1 or success2 or success3 or success4 or success5, {
            "get_compliance": response1,
            "post_compliance": response2,
            "contract_check": response3,
            "alternative_path": response4,
            "general_compliance": response5
        }

    def print_detailed_error_analysis(self):
        """Print detailed analysis of all errors encountered"""
        if not self.detailed_errors:
            print("\n✅ No detailed errors to analyze - all tests passed or had expected failures")
            return
            
        print("\n" + "="*80)
        print("🔍 DETAILED ERROR ANALYSIS")
        print("="*80)
        
        for i, error in enumerate(self.detailed_errors, 1):
            print(f"\n❌ ERROR {i}: {error['test_name']}")
            print("-" * 60)
            
            if 'expected_status' in error:
                print(f"Expected Status: {error['expected_status']}")
                print(f"Actual Status: {error['actual_status']}")
            
            print(f"URL: {error['url']}")
            print(f"Method: {error['method']}")
            
            if error.get('request_data'):
                print(f"Request Data:")
                print(json.dumps(error['request_data'], indent=2))
            
            if error.get('response_data'):
                print(f"Response Data:")
                if isinstance(error['response_data'], dict):
                    print(json.dumps(error['response_data'], indent=2))
                else:
                    print(error['response_data'])
            
            if error.get('error_analysis'):
                print(f"Analysis:")
                for analysis_point in error['error_analysis']:
                    print(f"  • {analysis_point}")
            
            if error.get('error_message'):
                print(f"Error Message: {error['error_message']}")

    def run_focused_hr_tests(self):
        """Run the 3 specific HR endpoint tests requested"""
        print("🚀 Starting Focused HR Endpoint Testing...")
        print("Testing 3 specific HR endpoints as requested:")
        print("1. HR Onboarding Workflow: POST /api/hr/onboarding/create")
        print("2. Contract Wizard Field Suggestions: POST /api/contract-wizard/suggestions") 
        print("3. HR Compliance: GET/POST /api/hr/compliance")
        print("=" * 80)
        
        # Test the 3 specific endpoints
        test1_success, test1_response = self.test_hr_onboarding_workflow_create()
        test2_success, test2_response = self.test_contract_wizard_field_suggestions()
        test3_success, test3_response = self.test_hr_compliance_endpoint()
        
        # Print detailed error analysis
        self.print_detailed_error_analysis()
        
        # Print summary
        print("\n" + "=" * 80)
        print("🎯 FOCUSED HR ENDPOINT TEST SUMMARY")
        print("=" * 80)
        print(f"Total Tests Run: {self.tests_run}")
        print(f"Tests Passed: {self.tests_passed}")
        print(f"Tests Failed: {self.tests_run - self.tests_passed}")
        print(f"Success Rate: {(self.tests_passed/self.tests_run)*100:.1f}%" if self.tests_run > 0 else "No tests run")
        
        # Specific endpoint results
        print(f"\n📊 ENDPOINT-SPECIFIC RESULTS:")
        print(f"1. HR Onboarding Workflow: {'✅ WORKING' if test1_success else '❌ ISSUES FOUND'}")
        print(f"2. Contract Wizard Suggestions: {'✅ WORKING' if test2_success else '❌ ISSUES FOUND'}")
        print(f"3. HR Compliance: {'✅ WORKING' if test3_success else '❌ ISSUES FOUND'}")
        
        # Detailed findings
        print(f"\n📋 DETAILED FINDINGS:")
        
        if not test1_success:
            print(f"❌ HR Onboarding Workflow Issues:")
            print(f"   - Check if workflow_type field is required")
            print(f"   - Verify parameter structure matches backend expectations")
            print(f"   - May need to implement endpoint if returning 404")
            
        if not test2_success:
            print(f"❌ Contract Wizard Suggestions Issues:")
            print(f"   - Parameter format may need adjustment")
            print(f"   - Check required vs optional fields")
            print(f"   - Verify request structure matches backend model")
            
        if not test3_success:
            print(f"❌ HR Compliance Issues:")
            print(f"   - Endpoint may not be implemented yet")
            print(f"   - Consider implementing GET/POST /api/hr/compliance")
            print(f"   - Alternative: use general compliance-check for HR content")
        
        return self.tests_passed, self.tests_run, {
            "onboarding": test1_response,
            "suggestions": test2_response, 
            "compliance": test3_response
        }

if __name__ == "__main__":
    tester = HRSpecificEndpointTester()
    passed, total, responses = tester.run_focused_hr_tests()
    
    # Exit with appropriate code
    if passed == total:
        sys.exit(0)  # All tests passed
    else:
        sys.exit(1)  # Some tests failed