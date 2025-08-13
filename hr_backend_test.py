import requests
import sys
import json
import uuid
from datetime import datetime, timedelta

class HRBackendTester:
    def __init__(self, base_url="https://legal-engine-check.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.employee_id = None
        self.policy_id = None
        self.workflow_id = None
        self.company_id = None

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

    def test_hr_contract_types_availability(self):
        """Test that HR-specific contract types are available in GET /api/contract-types"""
        success, response = self.run_test("HR Contract Types Availability", "GET", "contract-types", 200)
        
        if success and 'types' in response:
            types = response['types']
            print(f"   Found {len(types)} total contract types")
            
            # Check for HR-specific contract types
            hr_contract_types = [
                'offer_letter',
                'employee_handbook_acknowledgment',
                'severance_agreement', 
                'contractor_agreement',
                'employee_nda',
                'performance_improvement_plan',
                'employment_agreement'
            ]
            
            found_hr_types = []
            missing_hr_types = []
            
            type_ids = [t.get('id') for t in types]
            
            for hr_type in hr_contract_types:
                if hr_type in type_ids:
                    found_hr_types.append(hr_type)
                else:
                    missing_hr_types.append(hr_type)
            
            print(f"   ✅ Found HR contract types: {found_hr_types}")
            if missing_hr_types:
                print(f"   ❌ Missing HR contract types: {missing_hr_types}")
                return False, response
            else:
                print(f"   ✅ All required HR contract types are available")
                
            # Check for HR category
            categories = response.get('categories', [])
            hr_categories = [cat for cat in categories if 'hr' in cat.lower() or 'employment' in cat.lower()]
            if hr_categories:
                print(f"   ✅ Found HR/Employment categories: {hr_categories}")
            else:
                print(f"   ⚠️  No specific HR/Employment categories found")
                
        return success, response

    def test_hr_contract_generation_employment_agreement(self):
        """Test HR-specific contract generation - Employment Agreement"""
        employment_data = {
            "contract_type": "employment_agreement",
            "jurisdiction": "US",
            "parties": {
                "party1_name": "TechCorp Industries Inc.",
                "party1_type": "corporation",
                "party2_name": "Sarah Johnson",
                "party2_type": "individual"
            },
            "terms": {
                "position": "Senior Software Engineer",
                "department": "Engineering",
                "salary": "$95,000",
                "employment_type": "full_time",
                "start_date": "2025-02-01",
                "benefits_eligible": True,
                "work_schedule": "Monday-Friday, 9 AM - 5 PM",
                "location": "hybrid",
                "probationary_period": "90 days",
                "vacation_days": "15 days annually"
            },
            "special_clauses": ["Confidentiality agreement", "Non-compete clause for 6 months"]
        }
        
        success, response = self.run_test(
            "HR Employment Agreement Generation", 
            "POST", 
            "generate-contract", 
            200, 
            employment_data,
            timeout=60
        )
        
        if success and 'contract' in response:
            contract = response['contract']
            content = contract.get('content', '')
            
            print(f"   Contract ID: {contract.get('id')}")
            print(f"   Compliance Score: {contract.get('compliance_score', 'N/A')}%")
            print(f"   Content Length: {len(content)} characters")
            
            # Verify HR-specific content
            hr_keywords = ['employment', 'salary', 'benefits', 'position', 'probationary']
            found_keywords = [kw for kw in hr_keywords if kw.lower() in content.lower()]
            print(f"   ✅ Found HR keywords: {found_keywords}")
            
            if len(found_keywords) >= 3:
                print(f"   ✅ Employment agreement contains appropriate HR content")
            else:
                print(f"   ❌ Employment agreement may be missing HR-specific content")
                
        return success, response

    def test_hr_contract_generation_offer_letter(self):
        """Test HR-specific contract generation - Offer Letter"""
        offer_letter_data = {
            "contract_type": "offer_letter",
            "jurisdiction": "US",
            "parties": {
                "party1_name": "Innovation Labs LLC",
                "party1_type": "llc",
                "party2_name": "Michael Chen",
                "party2_type": "individual"
            },
            "terms": {
                "position": "Product Manager",
                "department": "Product Development",
                "salary": "$85,000",
                "start_date": "2025-03-01",
                "reporting_manager": "Director of Product",
                "employment_type": "full_time",
                "benefits_package": "Health, dental, vision insurance, 401k matching",
                "equity_options": "Stock options available after 1 year",
                "acceptance_deadline": "2025-01-31"
            },
            "special_clauses": ["Background check required", "At-will employment"]
        }
        
        success, response = self.run_test(
            "HR Offer Letter Generation", 
            "POST", 
            "generate-contract", 
            200, 
            offer_letter_data,
            timeout=60
        )
        
        if success and 'contract' in response:
            contract = response['contract']
            content = contract.get('content', '')
            
            print(f"   Contract ID: {contract.get('id')}")
            print(f"   Content Length: {len(content)} characters")
            
            # Verify offer letter specific content
            offer_keywords = ['offer', 'position', 'salary', 'start date', 'benefits', 'acceptance']
            found_keywords = [kw for kw in offer_keywords if kw.lower() in content.lower()]
            print(f"   ✅ Found offer letter keywords: {found_keywords}")
            
            if len(found_keywords) >= 4:
                print(f"   ✅ Offer letter contains appropriate content")
            else:
                print(f"   ❌ Offer letter may be missing key components")
                
        return success, response

    def test_hr_contract_generation_contractor_agreement(self):
        """Test HR-specific contract generation - Contractor Agreement"""
        contractor_data = {
            "contract_type": "contractor_agreement",
            "jurisdiction": "US",
            "parties": {
                "party1_name": "Digital Solutions Corp",
                "party1_type": "corporation",
                "party2_name": "Alex Rodriguez",
                "party2_type": "individual"
            },
            "terms": {
                "services": "Web development and UI/UX design services",
                "hourly_rate": "$75",
                "payment_terms": "Net 30 days",
                "contract_duration": "6 months",
                "work_location": "remote",
                "deliverables": "Responsive website with admin dashboard",
                "intellectual_property": "Work for hire - all rights to client",
                "tax_classification": "1099 contractor"
            },
            "special_clauses": ["Independent contractor status", "Equipment provided by contractor"]
        }
        
        success, response = self.run_test(
            "HR Contractor Agreement Generation", 
            "POST", 
            "generate-contract", 
            200, 
            contractor_data,
            timeout=60
        )
        
        if success and 'contract' in response:
            contract = response['contract']
            content = contract.get('content', '')
            
            print(f"   Contract ID: {contract.get('id')}")
            print(f"   Content Length: {len(content)} characters")
            
            # Verify contractor agreement specific content
            contractor_keywords = ['contractor', 'independent', 'services', 'hourly', '1099', 'deliverables']
            found_keywords = [kw for kw in contractor_keywords if kw.lower() in content.lower()]
            print(f"   ✅ Found contractor keywords: {found_keywords}")
            
            if len(found_keywords) >= 4:
                print(f"   ✅ Contractor agreement contains appropriate content")
            else:
                print(f"   ❌ Contractor agreement may be missing key components")
                
        return success, response

    def test_hr_employee_profile_creation(self):
        """Test POST /api/hr/employees - Create employee profile"""
        # First create a company for the employee
        company_data = {
            "name": "HR Test Company Inc.",
            "industry": "Technology",
            "size": "medium",
            "legal_structure": "corporation",
            "address": {
                "street": "123 Business Ave",
                "city": "San Francisco",
                "state": "CA",
                "country": "US",
                "zip": "94105"
            },
            "phone": "+1-555-0123",
            "email": "hr@hrtestcompany.com",
            "user_id": str(uuid.uuid4())
        }
        
        # Create company first
        company_success, company_response = self.run_test(
            "Create Company for HR Employee Test", 
            "POST", 
            "companies/profile", 
            200, 
            company_data
        )
        
        if company_success and 'id' in company_response:
            self.company_id = company_response['id']
            print(f"   Created company ID: {self.company_id}")
        else:
            print("   ⚠️  Could not create company, using placeholder ID")
            self.company_id = str(uuid.uuid4())
        
        # Now create employee profile
        employee_data = {
            "employee_id": "EMP001",
            "first_name": "Jennifer",
            "last_name": "Smith",
            "email": "jennifer.smith@hrtestcompany.com",
            "phone": "+1-555-0124",
            "department": "Engineering",
            "position": "Senior Developer",
            "employment_type": "full_time",
            "start_date": "2025-01-15T00:00:00Z",
            "manager_id": "MGR001",
            "salary": 95000.00,
            "benefits_eligible": True,
            "location": "hybrid",
            "employment_status": "active",
            "company_id": self.company_id
        }
        
        success, response = self.run_test(
            "HR Employee Profile Creation", 
            "POST", 
            "hr/employees", 
            200, 
            employee_data
        )
        
        if success and 'id' in response:
            self.employee_id = response['id']
            print(f"   Created employee ID: {self.employee_id}")
            
            # Verify response structure
            expected_fields = ['id', 'employee_id', 'first_name', 'last_name', 'email', 'department', 'position']
            missing_fields = [field for field in expected_fields if field not in response]
            
            if not missing_fields:
                print(f"   ✅ Employee profile has all required fields")
            else:
                print(f"   ❌ Missing fields in employee profile: {missing_fields}")
                
            # Verify data integrity
            if response.get('employment_type') == 'full_time':
                print(f"   ✅ Employment type correctly set")
            if response.get('benefits_eligible') == True:
                print(f"   ✅ Benefits eligibility correctly set")
                
        return success, response

    def test_hr_policy_creation(self):
        """Test POST /api/hr/policies - Create HR policy"""
        policy_data = {
            "title": "Remote Work Policy",
            "category": "employee_handbook",
            "content": """
            **REMOTE WORK POLICY**
            
            **1. PURPOSE**
            This policy establishes guidelines for remote work arrangements to ensure productivity, communication, and work-life balance.
            
            **2. ELIGIBILITY**
            Full-time employees with at least 6 months of employment may request remote work arrangements.
            
            **3. REQUIREMENTS**
            - Reliable internet connection
            - Dedicated workspace
            - Availability during core business hours (9 AM - 3 PM PST)
            - Regular check-ins with supervisor
            
            **4. APPROVAL PROCESS**
            Remote work requests must be submitted 30 days in advance and approved by direct supervisor and HR.
            """,
            "version": "1.0",
            "effective_date": "2025-02-01T00:00:00Z",
            "mandatory_acknowledgment": True,
            "applies_to": ["all"],
            "company_id": self.company_id or str(uuid.uuid4()),
            "created_by": "HR Manager",
            "status": "draft"
        }
        
        success, response = self.run_test(
            "HR Policy Creation", 
            "POST", 
            "hr/policies", 
            200, 
            policy_data
        )
        
        if success and 'id' in response:
            self.policy_id = response['id']
            print(f"   Created policy ID: {self.policy_id}")
            
            # Verify response structure
            expected_fields = ['id', 'title', 'category', 'content', 'version', 'effective_date', 'status']
            missing_fields = [field for field in expected_fields if field not in response]
            
            if not missing_fields:
                print(f"   ✅ HR policy has all required fields")
            else:
                print(f"   ❌ Missing fields in HR policy: {missing_fields}")
                
            # Verify policy content
            if 'Remote Work Policy' in response.get('title', ''):
                print(f"   ✅ Policy title correctly set")
            if response.get('category') == 'employee_handbook':
                print(f"   ✅ Policy category correctly set")
            if response.get('mandatory_acknowledgment') == True:
                print(f"   ✅ Mandatory acknowledgment correctly set")
                
        return success, response

    def test_hr_policy_templates(self):
        """Test GET /api/hr/policies/templates - Get policy templates"""
        success, response = self.run_test(
            "HR Policy Templates", 
            "GET", 
            "hr/policies/templates", 
            200
        )
        
        if success:
            if isinstance(response, dict) and 'templates' in response:
                templates = response['templates']
                print(f"   Found {len(templates)} policy templates")
                
                # Check for common HR policy templates
                expected_templates = [
                    'employee_handbook',
                    'code_of_conduct', 
                    'remote_work_policy',
                    'pto_policy',
                    'harassment_policy'
                ]
                
                template_names = [t.get('name', '').lower() for t in templates] if isinstance(templates, list) else []
                found_templates = []
                
                for expected in expected_templates:
                    if any(expected in name for name in template_names):
                        found_templates.append(expected)
                
                print(f"   ✅ Found expected templates: {found_templates}")
                
                if len(found_templates) >= 3:
                    print(f"   ✅ Sufficient policy templates available")
                else:
                    print(f"   ⚠️  Limited policy templates available")
                    
            elif isinstance(response, list):
                print(f"   Found {len(response)} policy templates (list format)")
                print(f"   ✅ Policy templates endpoint working")
            else:
                print(f"   ⚠️  Unexpected response format for policy templates")
                
        return success, response

    def test_hr_smart_suggestions(self):
        """Test POST /api/hr/suggestions - Get HR smart suggestions"""
        suggestions_request = {
            "field_name": "salary",
            "position": "Software Engineer",
            "department": "Engineering",
            "location": "San Francisco, CA",
            "experience_level": "senior",
            "company_size": "medium",
            "industry": "Technology"
        }
        
        success, response = self.run_test(
            "HR Smart Suggestions - Salary", 
            "POST", 
            "hr/suggestions", 
            200, 
            suggestions_request
        )
        
        if success:
            if isinstance(response, dict) and 'suggestions' in response:
                suggestions = response['suggestions']
                print(f"   Received {len(suggestions)} salary suggestions")
                
                # Verify suggestion structure
                if suggestions and isinstance(suggestions, list):
                    first_suggestion = suggestions[0]
                    expected_fields = ['field_name', 'suggested_value', 'confidence', 'reasoning', 'source']
                    missing_fields = [field for field in expected_fields if field not in first_suggestion]
                    
                    if not missing_fields:
                        print(f"   ✅ Suggestion structure is correct")
                        print(f"   Suggested salary: {first_suggestion.get('suggested_value')}")
                        print(f"   Confidence: {first_suggestion.get('confidence')}")
                    else:
                        print(f"   ❌ Missing fields in suggestion: {missing_fields}")
                        
            else:
                print(f"   ⚠️  Unexpected response format for HR suggestions")
        
        # Test employment type suggestions
        employment_type_request = {
            "field_name": "employment_type",
            "position": "Marketing Coordinator",
            "department": "Marketing",
            "company_size": "small",
            "industry": "Technology"
        }
        
        success2, response2 = self.run_test(
            "HR Smart Suggestions - Employment Type", 
            "POST", 
            "hr/suggestions", 
            200, 
            employment_type_request
        )
        
        if success2:
            print(f"   ✅ Employment type suggestions working")
            
        # Test benefits suggestions
        benefits_request = {
            "field_name": "benefits_eligible",
            "employment_type": "full_time",
            "position": "Senior Developer",
            "company_size": "medium"
        }
        
        success3, response3 = self.run_test(
            "HR Smart Suggestions - Benefits", 
            "POST", 
            "hr/suggestions", 
            200, 
            benefits_request
        )
        
        if success3:
            print(f"   ✅ Benefits eligibility suggestions working")
            
        return success and success2 and success3, {
            "salary": response,
            "employment_type": response2, 
            "benefits": response3
        }

    def test_hr_onboarding_workflow_creation(self):
        """Test POST /api/hr/onboarding/create - Create onboarding workflow"""
        if not self.employee_id:
            print("   ⚠️  No employee ID available, creating placeholder")
            self.employee_id = str(uuid.uuid4())
            
        onboarding_data = {
            "employee_id": self.employee_id,
            "workflow_template": "standard",
            "workflow_options": {
                "include_it_setup": True,
                "include_benefits_enrollment": True,
                "include_policy_acknowledgment": True,
                "assigned_buddy": "senior_team_member",
                "department_specific_training": True
            },
            "company_id": self.company_id or str(uuid.uuid4())
        }
        
        success, response = self.run_test(
            "HR Onboarding Workflow Creation", 
            "POST", 
            "hr/onboarding/create", 
            200, 
            onboarding_data
        )
        
        if success and 'workflow_id' in response:
            self.workflow_id = response['workflow_id']
            print(f"   Created workflow ID: {self.workflow_id}")
            
            # Verify response structure
            expected_fields = ['workflow_id', 'status', 'current_step', 'next_actions', 'estimated_completion']
            missing_fields = [field for field in expected_fields if field not in response]
            
            if not missing_fields:
                print(f"   ✅ Onboarding workflow has all required fields")
            else:
                print(f"   ❌ Missing fields in workflow response: {missing_fields}")
                
            # Verify workflow details
            if response.get('status') in ['pending', 'in_progress']:
                print(f"   ✅ Workflow status is appropriate: {response.get('status')}")
                
            current_step = response.get('current_step', {})
            if isinstance(current_step, dict) and current_step:
                print(f"   ✅ Current step information provided")
                
            next_actions = response.get('next_actions', [])
            if isinstance(next_actions, list) and next_actions:
                print(f"   ✅ Next actions provided: {len(next_actions)} actions")
                
        return success, response

    def test_hr_contract_wizard_integration(self):
        """Test that contract wizard shows HR-specific steps for HR contract types"""
        # Test contract wizard initialization with HR contract type
        wizard_request = {
            "contract_type": "employment_agreement",
            "current_step": 1,
            "partial_data": {
                "industry": "Technology",
                "company_size": "medium"
            }
        }
        
        success, response = self.run_test(
            "HR Contract Wizard Integration", 
            "POST", 
            "contract-wizard/initialize", 
            200, 
            wizard_request
        )
        
        if success:
            # Verify wizard response structure
            expected_fields = ['current_step', 'suggestions', 'progress', 'estimated_completion_time']
            missing_fields = [field for field in expected_fields if field not in response]
            
            if not missing_fields:
                print(f"   ✅ Contract wizard response structure is correct")
            else:
                print(f"   ❌ Missing fields in wizard response: {missing_fields}")
                
            # Check for HR-specific suggestions
            suggestions = response.get('suggestions', [])
            if suggestions:
                hr_related_suggestions = []
                for suggestion in suggestions:
                    suggestion_text = str(suggestion).lower()
                    if any(hr_term in suggestion_text for hr_term in ['employment', 'salary', 'benefits', 'position', 'department']):
                        hr_related_suggestions.append(suggestion)
                        
                if hr_related_suggestions:
                    print(f"   ✅ Found {len(hr_related_suggestions)} HR-related suggestions")
                else:
                    print(f"   ⚠️  No HR-specific suggestions found")
                    
            # Test field-specific suggestions for HR fields
            hr_field_request = {
                "field_name": "employment_type",
                "contract_type": "employment_agreement",
                "context": {
                    "position": "Software Engineer",
                    "department": "Engineering"
                }
            }
            
            success2, response2 = self.run_test(
                "HR Contract Wizard Field Suggestions", 
                "POST", 
                "contract-wizard/suggestions", 
                200, 
                hr_field_request
            )
            
            if success2:
                print(f"   ✅ HR field suggestions working")
                
        return success, response

    def test_hr_compliance_tracking(self):
        """Test HR compliance tracking functionality"""
        # This tests if the system can handle HR compliance requirements
        compliance_data = {
            "company_id": self.company_id or str(uuid.uuid4()),
            "compliance_area": "employment_law",
            "requirement": "I-9 Employment Eligibility Verification",
            "description": "Verify employment eligibility for all new hires within 3 days of start date",
            "jurisdiction": "US",
            "due_date": "2025-02-15T00:00:00Z",
            "status": "pending",
            "risk_level": "high",
            "documents_required": ["I-9 Form", "Acceptable identification documents"],
            "notes": "Required for all new employees starting after January 1, 2025"
        }
        
        # Try to create compliance tracker (this endpoint might not exist yet)
        success, response = self.run_test(
            "HR Compliance Tracking", 
            "POST", 
            "hr/compliance", 
            200, 
            compliance_data
        )
        
        if not success:
            # If endpoint doesn't exist, try alternative approach
            print("   ⚠️  Direct compliance endpoint not available, testing via contract analysis")
            
            # Test compliance through contract analysis
            employment_contract_content = """
            EMPLOYMENT AGREEMENT
            
            This Employment Agreement is entered into between TechCorp Inc. and John Doe.
            
            Position: Software Engineer
            Salary: $90,000 annually
            Benefits: Health insurance, 401k matching
            Employment Type: Full-time, at-will
            """
            
            compliance_check_data = {
                "contract_content": employment_contract_content,
                "jurisdictions": ["US"]
            }
            
            success_alt, response_alt = self.run_test(
                "HR Compliance via Contract Analysis", 
                "POST", 
                "compliance-check", 
                200, 
                compliance_check_data
            )
            
            if success_alt:
                print(f"   ✅ HR compliance checking available via contract analysis")
                return True, response_alt
                
        return success, response

    def run_all_hr_tests(self):
        """Run all HR & Employment backend tests"""
        print("🚀 Starting HR & Employment Backend Testing...")
        print("=" * 60)
        
        # Test 1: HR Contract Types Availability
        self.test_hr_contract_types_availability()
        
        # Test 2: HR Contract Generation
        self.test_hr_contract_generation_employment_agreement()
        self.test_hr_contract_generation_offer_letter()
        self.test_hr_contract_generation_contractor_agreement()
        
        # Test 3: HR API Endpoints
        self.test_hr_employee_profile_creation()
        self.test_hr_policy_creation()
        self.test_hr_policy_templates()
        self.test_hr_smart_suggestions()
        self.test_hr_onboarding_workflow_creation()
        
        # Test 4: HR Contract Wizard Integration
        self.test_hr_contract_wizard_integration()
        
        # Test 5: HR Compliance
        self.test_hr_compliance_tracking()
        
        # Print summary
        print("\n" + "=" * 60)
        print("🎯 HR & EMPLOYMENT BACKEND TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests Run: {self.tests_run}")
        print(f"Tests Passed: {self.tests_passed}")
        print(f"Tests Failed: {self.tests_run - self.tests_passed}")
        print(f"Success Rate: {(self.tests_passed/self.tests_run)*100:.1f}%" if self.tests_run > 0 else "No tests run")
        
        if self.tests_passed == self.tests_run:
            print("🎉 ALL HR BACKEND TESTS PASSED!")
        elif self.tests_passed > self.tests_run * 0.8:
            print("✅ MOST HR BACKEND TESTS PASSED - Minor issues detected")
        else:
            print("❌ SIGNIFICANT HR BACKEND ISSUES DETECTED")
            
        return self.tests_passed, self.tests_run

if __name__ == "__main__":
    tester = HRBackendTester()
    passed, total = tester.run_all_hr_tests()
    
    # Exit with appropriate code
    if passed == total:
        sys.exit(0)  # All tests passed
    else:
        sys.exit(1)  # Some tests failed