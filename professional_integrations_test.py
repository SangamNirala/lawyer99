import requests
import sys
import json
import uuid
from datetime import datetime

class ProfessionalIntegrationsAPITester:
    def __init__(self, base_url="https://9c72c586-5126-47dc-85cc-7afca9190d08.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.generated_api_key = None
        self.workflow_id = None
        self.app_id = None

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

    # PROFESSIONAL INTEGRATIONS FRAMEWORK TESTS (3 endpoints)
    def test_integrations_status(self):
        """Test GET /api/integrations/status - Integration status for 15+ legal software integrations"""
        success, response = self.run_test(
            "Professional Integrations Status", 
            "GET", 
            "integrations/status", 
            200
        )
        
        if success and isinstance(response, dict):
            print(f"   Checking integration status response structure...")
            
            # Check for expected fields
            expected_fields = ['integrations', 'total_integrations', 'active_integrations']
            for field in expected_fields:
                if field in response:
                    print(f"   ✅ Found expected field: {field}")
                else:
                    print(f"   ⚠️  Missing expected field: {field}")
            
            # Check integrations list
            if 'integrations' in response and isinstance(response['integrations'], list):
                integrations = response['integrations']
                print(f"   Found {len(integrations)} integrations")
                
                # Look for expected integration types
                expected_integrations = ['EspoCRM', 'SuiteCRM', 'Google Drive', 'Dropbox', 'GitHub', 'NextCloud', 'CourtListener']
                found_integrations = [i.get('name', '') for i in integrations]
                
                for expected in expected_integrations:
                    if any(expected.lower() in found.lower() for found in found_integrations):
                        print(f"   ✅ Found expected integration: {expected}")
                    else:
                        print(f"   ⚠️  Missing expected integration: {expected}")
                
                # Check integration structure
                if integrations:
                    sample_integration = integrations[0]
                    integration_fields = ['name', 'type', 'status', 'provider']
                    for field in integration_fields:
                        if field in sample_integration:
                            print(f"   ✅ Integration has field: {field}")
                        else:
                            print(f"   ⚠️  Integration missing field: {field}")
        
        return success, response

    def test_integrations_activate(self):
        """Test POST /api/integrations/activate - Activate specific integrations with config overrides"""
        activation_data = {
            "integration_id": "google_drive",
            "config_overrides": {
                "folder_path": "/legal_documents",
                "sync_enabled": True,
                "access_level": "read_write"
            }
        }
        
        success, response = self.run_test(
            "Integration Activation", 
            "POST", 
            "integrations/activate", 
            200, 
            activation_data
        )
        
        if success and isinstance(response, dict):
            print(f"   Checking activation response structure...")
            
            # Check for expected fields
            expected_fields = ['success', 'integration_name', 'status', 'capabilities']
            for field in expected_fields:
                if field in response:
                    print(f"   ✅ Found expected field: {field}")
                else:
                    print(f"   ⚠️  Missing expected field: {field}")
            
            # Check activation success
            if response.get('success'):
                print(f"   ✅ Integration activation reported as successful")
            else:
                print(f"   ⚠️  Integration activation may have failed")
        
        return success, response

    def test_integrations_action(self):
        """Test POST /api/integrations/action - Execute actions using integrations"""
        action_data = {
            "integration_id": "courtlistener",
            "action": "search_cases",
            "parameters": {
                "query": "contract dispute",
                "jurisdiction": "US",
                "max_results": 10
            }
        }
        
        success, response = self.run_test(
            "Integration Action Execution", 
            "POST", 
            "integrations/action", 
            200, 
            action_data
        )
        
        if success and isinstance(response, dict):
            print(f"   Checking action execution response structure...")
            
            # Check for expected fields
            expected_fields = ['success', 'action_type', 'results', 'execution_time']
            for field in expected_fields:
                if field in response:
                    print(f"   ✅ Found expected field: {field}")
                else:
                    print(f"   ⚠️  Missing expected field: {field}")
            
            # Check results
            if 'results' in response:
                results = response['results']
                print(f"   Action returned {len(results) if isinstance(results, list) else 'non-list'} results")
        
        return success, response

    # PROFESSIONAL API ECOSYSTEM TESTS (3 endpoints)
    def test_api_ecosystem_generate_key(self):
        """Test POST /api/api-ecosystem/generate-key - Generate professional API keys"""
        key_data = {
            "name": "Test Legal Firm API Key",
            "description": "API key for contract analysis and legal research automation",
            "access_level": "professional",
            "organization_id": "test-org-123",
            "law_firm_name": "Test Legal Firm LLC",
            "expires_in_days": 365,
            "rate_limits": {
                "requests_per_minute": 100,
                "requests_per_day": 1000
            }
        }
        
        success, response = self.run_test(
            "API Key Generation", 
            "POST", 
            "api-ecosystem/generate-key", 
            200, 
            key_data
        )
        
        if success and isinstance(response, dict):
            print(f"   Checking API key generation response structure...")
            
            # Check for expected fields
            expected_fields = ['api_key', 'access_level', 'rate_limit', 'expires_at', 'organization_id']
            for field in expected_fields:
                if field in response:
                    print(f"   ✅ Found expected field: {field}")
                    if field == 'api_key':
                        self.generated_api_key = response[field]
                        # Fix: Ensure api_key is a string before slicing
                        if isinstance(self.generated_api_key, str) and len(self.generated_api_key) > 20:
                            print(f"   Generated API key: {self.generated_api_key[:20]}...")
                        else:
                            print(f"   Generated API key: {self.generated_api_key}")
                else:
                    print(f"   ⚠️  Missing expected field: {field}")
            
            # Validate API key format
            if 'api_key' in response:
                api_key = response['api_key']
                if len(api_key) >= 32:  # Reasonable API key length
                    print(f"   ✅ API key has reasonable length: {len(api_key)} characters")
                else:
                    print(f"   ⚠️  API key seems too short: {len(api_key)} characters")
        
        return success, response

    def test_api_ecosystem_documentation(self):
        """Test GET /api/api-ecosystem/documentation - API documentation"""
        success, response = self.run_test(
            "API Documentation", 
            "GET", 
            "api-ecosystem/documentation", 
            200
        )
        
        if success and isinstance(response, dict):
            print(f"   Checking API documentation response structure...")
            
            # Check for expected fields
            expected_fields = ['authentication', 'rate_limits', 'endpoints', 'response_schemas', 'integration_examples']
            for field in expected_fields:
                if field in response:
                    print(f"   ✅ Found expected field: {field}")
                else:
                    print(f"   ⚠️  Missing expected field: {field}")
            
            # Check endpoints documentation
            if 'endpoints' in response and isinstance(response['endpoints'], dict):
                endpoints = response['endpoints']
                print(f"   Found documentation for {len(endpoints)} endpoint categories")
                
                # Look for expected endpoint categories
                expected_categories = ['contract_generation', 'legal_research', 'compliance_checking']
                for category in expected_categories:
                    if category in endpoints:
                        print(f"   ✅ Found endpoint category: {category}")
                    else:
                        print(f"   ⚠️  Missing endpoint category: {category}")
        
        return success, response

    def test_api_ecosystem_usage_analytics(self):
        """Test GET /api/api-ecosystem/usage-analytics - API usage analytics"""
        # Test with query parameters
        endpoint = "api-ecosystem/usage-analytics?time_period=30d&organization_id=test-org-123"
        
        success, response = self.run_test(
            "API Usage Analytics", 
            "GET", 
            endpoint, 
            200
        )
        
        if success and isinstance(response, dict):
            print(f"   Checking usage analytics response structure...")
            
            # Check for expected fields
            expected_fields = ['time_period', 'total_requests', 'success_rate', 'average_response_time', 'usage_by_endpoint']
            for field in expected_fields:
                if field in response:
                    print(f"   ✅ Found expected field: {field}")
                else:
                    print(f"   ⚠️  Missing expected field: {field}")
            
            # Check analytics data
            if 'total_requests' in response:
                total_requests = response['total_requests']
                print(f"   Total API requests: {total_requests}")
            
            if 'success_rate' in response:
                success_rate = response['success_rate']
                print(f"   API success rate: {success_rate}%")
        
        return success, response

    # ENTERPRISE INTEGRATION FEATURES TESTS (3 endpoints)
    def test_enterprise_sso_authenticate(self):
        """Test POST /api/enterprise/sso/authenticate - SSO authentication with 6 providers"""
        sso_data = {
            "provider_id": "google_workspace",
            "auth_code": "mock-auth-code-12345",
            "state": "test-state-token"
        }
        
        success, response = self.run_test(
            "Enterprise SSO Authentication", 
            "POST", 
            "enterprise/sso/authenticate", 
            200, 
            sso_data
        )
        
        if success and isinstance(response, dict):
            print(f"   Checking SSO authentication response structure...")
            
            # Check for expected fields
            expected_fields = ['success', 'user_id', 'session_token', 'provider', 'organization_id']
            for field in expected_fields:
                if field in response:
                    print(f"   ✅ Found expected field: {field}")
                else:
                    print(f"   ⚠️  Missing expected field: {field}")
            
            # Check authentication success
            if response.get('success'):
                print(f"   ✅ SSO authentication reported as successful")
            else:
                print(f"   ⚠️  SSO authentication may have failed")
        
        return success, response

    def test_enterprise_compliance_check(self):
        """Test GET /api/enterprise/compliance/check - Compliance checking across frameworks"""
        # Test with query parameters for specific compliance frameworks
        endpoint = "enterprise/compliance/check?framework=SOC2&organization_id=test-org-123"
        
        success, response = self.run_test(
            "Enterprise Compliance Check", 
            "GET", 
            endpoint, 
            200
        )
        
        if success and isinstance(response, dict):
            print(f"   Checking compliance check response structure...")
            
            # Check for expected fields
            expected_fields = ['compliance_percentage', 'frameworks_checked', 'non_compliant_rules', 'recommendations']
            for field in expected_fields:
                if field in response:
                    print(f"   ✅ Found expected field: {field}")
                else:
                    print(f"   ⚠️  Missing expected field: {field}")
            
            # Check compliance data
            if 'compliance_percentage' in response:
                compliance_pct = response['compliance_percentage']
                print(f"   Overall compliance percentage: {compliance_pct}%")
            
            if 'frameworks_checked' in response:
                frameworks = response['frameworks_checked']
                print(f"   Frameworks checked: {frameworks}")
                
                # Look for expected frameworks
                expected_frameworks = ['SOC2', 'HIPAA', 'GDPR', 'ISO27001', 'Attorney-Client Privilege']
                for framework in expected_frameworks:
                    if framework in str(frameworks):
                        print(f"   ✅ Found expected framework: {framework}")
        
        return success, response

    def test_enterprise_audit_trail(self):
        """Test GET /api/enterprise/audit-trail - Audit trail for compliance"""
        # Test with query parameters
        endpoint = "enterprise/audit-trail?user_id=test-user-123&event_type=contract_generation&days=30"
        
        success, response = self.run_test(
            "Enterprise Audit Trail", 
            "GET", 
            endpoint, 
            200
        )
        
        if success and isinstance(response, dict):
            print(f"   Checking audit trail response structure...")
            
            # Check for expected fields
            expected_fields = ['events', 'total_events', 'time_range', 'event_summary']
            for field in expected_fields:
                if field in response:
                    print(f"   ✅ Found expected field: {field}")
                else:
                    print(f"   ⚠️  Missing expected field: {field}")
            
            # Check events structure
            if 'events' in response and isinstance(response['events'], list):
                events = response['events']
                print(f"   Found {len(events)} audit events")
                
                if events:
                    sample_event = events[0]
                    event_fields = ['event_id', 'timestamp', 'user_id', 'event_type', 'details']
                    for field in event_fields:
                        if field in sample_event:
                            print(f"   ✅ Event has field: {field}")
                        else:
                            print(f"   ⚠️  Event missing field: {field}")
        
        return success, response

    # LEGAL WORKFLOW AUTOMATION TESTS (5 endpoints)
    def test_workflows_templates(self):
        """Test GET /api/workflows/templates - 5 comprehensive workflow templates"""
        success, response = self.run_test(
            "Legal Workflow Templates", 
            "GET", 
            "workflows/templates", 
            200
        )
        
        if success and isinstance(response, dict):
            print(f"   Checking workflow templates response structure...")
            
            # Check for expected fields
            expected_fields = ['templates', 'total_templates']
            for field in expected_fields:
                if field in response:
                    print(f"   ✅ Found expected field: {field}")
                else:
                    print(f"   ⚠️  Missing expected field: {field}")
            
            # Check templates
            if 'templates' in response and isinstance(response['templates'], list):
                templates = response['templates']
                print(f"   Found {len(templates)} workflow templates")
                
                # Look for expected template types
                expected_templates = ['Client Onboarding', 'Contract Review', 'Legal Research', 'Document Generation', 'Case Management']
                found_templates = [t.get('name', '') for t in templates]
                
                for expected in expected_templates:
                    if any(expected.lower() in found.lower() for found in found_templates):
                        print(f"   ✅ Found expected template: {expected}")
                    else:
                        print(f"   ⚠️  Missing expected template: {expected}")
                
                # Check template structure
                if templates:
                    sample_template = templates[0]
                    template_fields = ['id', 'name', 'description', 'tasks', 'sla_hours', 'automation_level']
                    for field in template_fields:
                        if field in sample_template:
                            print(f"   ✅ Template has field: {field}")
                        else:
                            print(f"   ⚠️  Template missing field: {field}")
        
        return success, response

    def test_workflows_create(self):
        """Test POST /api/workflows/create - Create workflow from template"""
        workflow_data = {
            "template_id": "client_onboarding",
            "law_firm_id": "test-firm-123",
            "created_by": "jane.smith@testfirm.com",
            "client_id": "client-123",
            "matter_id": "MATTER-2025-001",
            "custom_parameters": {
                "client_type": "corporate",
                "practice_area": "contract_law",
                "priority": "high"
            }
        }
        
        success, response = self.run_test(
            "Workflow Creation", 
            "POST", 
            "workflows/create", 
            200, 
            workflow_data
        )
        
        if success and isinstance(response, dict):
            print(f"   Checking workflow creation response structure...")
            
            # Check for expected fields
            expected_fields = ['workflow_id', 'status', 'template_used', 'tasks_created', 'estimated_completion']
            for field in expected_fields:
                if field in response:
                    print(f"   ✅ Found expected field: {field}")
                    if field == 'workflow_id':
                        self.workflow_id = response[field]
                        print(f"   Created workflow ID: {self.workflow_id}")
                else:
                    print(f"   ⚠️  Missing expected field: {field}")
            
            # Check workflow creation success
            if response.get('status') == 'created':
                print(f"   ✅ Workflow creation reported as successful")
            else:
                print(f"   ⚠️  Workflow creation status: {response.get('status')}")
        
        return success, response

    def test_workflows_start(self):
        """Test POST /api/workflows/start - Start workflow execution"""
        if not self.workflow_id:
            print("⚠️  Skipping workflow start test - no workflow ID available")
            return True, {}
        
        start_data = {
            "workflow_id": self.workflow_id,
            "start_immediately": True,
            "notify_stakeholders": True
        }
        
        success, response = self.run_test(
            "Workflow Start Execution", 
            "POST", 
            "workflows/start", 
            200, 
            start_data
        )
        
        if success and isinstance(response, dict):
            print(f"   Checking workflow start response structure...")
            
            # Check for expected fields
            expected_fields = ['success', 'workflow_id', 'status', 'started_at', 'current_task']
            for field in expected_fields:
                if field in response:
                    print(f"   ✅ Found expected field: {field}")
                else:
                    print(f"   ⚠️  Missing expected field: {field}")
            
            # Check workflow start success
            if response.get('success'):
                print(f"   ✅ Workflow start reported as successful")
            else:
                print(f"   ⚠️  Workflow start may have failed")
        
        return success, response

    def test_workflows_status(self):
        """Test GET /api/workflows/{workflow_id}/status - Workflow status"""
        if not self.workflow_id:
            print("⚠️  Skipping workflow status test - no workflow ID available")
            return True, {}
        
        success, response = self.run_test(
            "Workflow Status Check", 
            "GET", 
            f"workflows/{self.workflow_id}/status", 
            200
        )
        
        if success and isinstance(response, dict):
            print(f"   Checking workflow status response structure...")
            
            # Check for expected fields
            expected_fields = ['workflow_id', 'status', 'progress_percentage', 'current_task', 'tasks_completed', 'estimated_completion']
            for field in expected_fields:
                if field in response:
                    print(f"   ✅ Found expected field: {field}")
                else:
                    print(f"   ⚠️  Missing expected field: {field}")
            
            # Check status data
            if 'progress_percentage' in response:
                progress = response['progress_percentage']
                print(f"   Workflow progress: {progress}%")
            
            if 'current_task' in response:
                current_task = response['current_task']
                print(f"   Current task: {current_task}")
        
        return success, response

    def test_workflows_analytics(self):
        """Test GET /api/workflows/analytics - Workflow analytics"""
        success, response = self.run_test(
            "Workflow Analytics", 
            "GET", 
            "workflows/analytics", 
            200
        )
        
        if success and isinstance(response, dict):
            print(f"   Checking workflow analytics response structure...")
            
            # Check for expected fields
            expected_fields = ['completion_rates', 'average_execution_times', 'success_rates_by_template', 'resource_utilization']
            for field in expected_fields:
                if field in response:
                    print(f"   ✅ Found expected field: {field}")
                else:
                    print(f"   ⚠️  Missing expected field: {field}")
            
            # Check analytics data
            if 'completion_rates' in response:
                completion_rates = response['completion_rates']
                print(f"   Workflow completion rates: {completion_rates}")
            
            if 'average_execution_times' in response:
                avg_times = response['average_execution_times']
                print(f"   Average execution times: {avg_times}")
        
        return success, response

    # MARKETPLACE & PARTNERSHIP ECOSYSTEM TESTS (7 endpoints)
    def test_marketplace_search(self):
        """Test POST /api/marketplace/search - Search marketplace apps"""
        search_data = {
            "category": "legal_research",
            "search_query": "legal research",
            "pricing_model": "subscription",
            "min_rating": 4.0,
            "tags": ["research", "ai"],
            "limit": 10
        }
        
        success, response = self.run_test(
            "Marketplace App Search", 
            "POST", 
            "marketplace/search", 
            200, 
            search_data
        )
        
        if success and isinstance(response, dict):
            print(f"   Checking marketplace search response structure...")
            
            # Check for expected fields
            expected_fields = ['apps', 'total_results', 'search_query', 'filters_applied']
            for field in expected_fields:
                if field in response:
                    print(f"   ✅ Found expected field: {field}")
                else:
                    print(f"   ⚠️  Missing expected field: {field}")
            
            # Check apps structure
            if 'apps' in response and isinstance(response['apps'], list):
                apps = response['apps']
                print(f"   Found {len(apps)} marketplace apps")
                
                if apps:
                    sample_app = apps[0]
                    self.app_id = sample_app.get('id')  # Store for later tests
                    app_fields = ['id', 'name', 'description', 'rating', 'install_count', 'pricing']
                    for field in app_fields:
                        if field in sample_app:
                            print(f"   ✅ App has field: {field}")
                        else:
                            print(f"   ⚠️  App missing field: {field}")
        
        return success, response

    def test_marketplace_app_details(self):
        """Test GET /api/marketplace/apps/{app_id} - App details"""
        if not self.app_id:
            # Use a default app ID for testing
            self.app_id = "legal-research-pro"
        
        success, response = self.run_test(
            "Marketplace App Details", 
            "GET", 
            f"marketplace/apps/{self.app_id}", 
            200
        )
        
        if success and isinstance(response, dict):
            print(f"   Checking app details response structure...")
            
            # Check for expected fields
            expected_fields = ['id', 'name', 'description', 'features', 'api_endpoints', 'pricing', 'security_standards', 'reviews']
            for field in expected_fields:
                if field in response:
                    print(f"   ✅ Found expected field: {field}")
                else:
                    print(f"   ⚠️  Missing expected field: {field}")
            
            # Check detailed information
            if 'features' in response and isinstance(response['features'], list):
                features = response['features']
                print(f"   App has {len(features)} features")
            
            if 'security_standards' in response:
                security = response['security_standards']
                print(f"   Security standards: {security}")
        
        return success, response

    def test_marketplace_install(self):
        """Test POST /api/marketplace/install - Install marketplace app"""
        if not self.app_id:
            self.app_id = "legal-research-pro"
        
        install_data = {
            "app_id": self.app_id,
            "user_id": "test-user-123",
            "law_firm_id": "test-firm-123",
            "installation_config": {
                "api_access_level": "full",
                "data_sharing": "limited",
                "notification_preferences": "email"
            }
        }
        
        success, response = self.run_test(
            "Marketplace App Installation", 
            "POST", 
            "marketplace/install", 
            200, 
            install_data
        )
        
        if success and isinstance(response, dict):
            print(f"   Checking app installation response structure...")
            
            # Check for expected fields
            expected_fields = ['success', 'installation_id', 'app_id', 'status', 'api_key', 'webhook_url']
            for field in expected_fields:
                if field in response:
                    print(f"   ✅ Found expected field: {field}")
                else:
                    print(f"   ⚠️  Missing expected field: {field}")
            
            # Check installation success
            if response.get('success'):
                print(f"   ✅ App installation reported as successful")
            else:
                print(f"   ⚠️  App installation may have failed")
        
        return success, response

    def test_marketplace_review(self):
        """Test POST /api/marketplace/review - Submit app review"""
        if not self.app_id:
            self.app_id = "legal-research-pro"
        
        review_data = {
            "app_id": self.app_id,
            "user_id": "test-user-123",
            "rating": 4,  # Changed to integer
            "title": "Excellent legal research capabilities",
            "review_text": "This app has significantly improved our legal research workflow. The AI-powered search is very accurate and the integration with our existing systems was seamless.",
            "law_firm_name": "Test Legal Firm LLC",
            "use_case": "Contract analysis and case law research",
            "pros": ["Fast search", "Accurate results", "Easy integration"],
            "cons": ["Could use more customization options"]
        }
        
        success, response = self.run_test(
            "Marketplace App Review Submission", 
            "POST", 
            "marketplace/review", 
            200, 
            review_data
        )
        
        if success and isinstance(response, dict):
            print(f"   Checking review submission response structure...")
            
            # Check for expected fields
            expected_fields = ['success', 'review_id', 'app_id', 'status', 'updated_rating']
            for field in expected_fields:
                if field in response:
                    print(f"   ✅ Found expected field: {field}")
                else:
                    print(f"   ⚠️  Missing expected field: {field}")
            
            # Check review submission success
            if response.get('success'):
                print(f"   ✅ Review submission reported as successful")
            else:
                print(f"   ⚠️  Review submission may have failed")
        
        return success, response

    def test_partnerships_apply(self):
        """Test POST /api/partnerships/apply - Partner application"""
        partnership_data = {
            "organization_name": "Legal Tech Innovations LLC",
            "partner_type": "Technology",
            "contact_name": "John Smith",
            "contact_email": "partnerships@legaltechinnovations.com",
            "business_info": {
                "contact_phone": "+1-555-0123",
                "business_description": "We develop AI-powered legal research and document analysis tools for law firms and corporate legal departments.",
                "integration_capabilities": ["API integration", "Webhook support", "SSO authentication"],
                "target_market": "Mid-size to large law firms",
                "geographic_regions": ["North America", "Europe"],
                "specializations": ["Contract analysis", "Legal research", "Compliance monitoring"]
            }
        }
        
        success, response = self.run_test(
            "Partnership Application", 
            "POST", 
            "partnerships/apply", 
            200, 
            partnership_data
        )
        
        if success and isinstance(response, dict):
            print(f"   Checking partnership application response structure...")
            
            # Check for expected fields
            expected_fields = ['success', 'application_id', 'status', 'next_steps', 'contact_timeline']
            for field in expected_fields:
                if field in response:
                    print(f"   ✅ Found expected field: {field}")
                else:
                    print(f"   ⚠️  Missing expected field: {field}")
            
            # Check application success
            if response.get('success'):
                print(f"   ✅ Partnership application reported as successful")
            else:
                print(f"   ⚠️  Partnership application may have failed")
        
        return success, response

    def test_partnerships_search(self):
        """Test GET /api/partnerships/search - Search partners"""
        # Test with query parameters
        endpoint = "partnerships/search?partner_type=Technology&region=North America&specialization=contract_analysis"
        
        success, response = self.run_test(
            "Partnership Search", 
            "GET", 
            endpoint, 
            200
        )
        
        if success and isinstance(response, dict):
            print(f"   Checking partnership search response structure...")
            
            # Check for expected fields
            expected_fields = ['partners', 'total_results', 'search_filters']
            for field in expected_fields:
                if field in response:
                    print(f"   ✅ Found expected field: {field}")
                else:
                    print(f"   ⚠️  Missing expected field: {field}")
            
            # Check partners structure
            if 'partners' in response and isinstance(response['partners'], list):
                partners = response['partners']
                print(f"   Found {len(partners)} partners")
                
                if partners:
                    sample_partner = partners[0]
                    partner_fields = ['id', 'name', 'partner_type', 'specializations', 'satisfaction_score', 'capabilities']
                    for field in partner_fields:
                        if field in sample_partner:
                            print(f"   ✅ Partner has field: {field}")
                        else:
                            print(f"   ⚠️  Partner missing field: {field}")
        
        return success, response

    def test_marketplace_analytics(self):
        """Test GET /api/marketplace/analytics - Marketplace analytics"""
        success, response = self.run_test(
            "Marketplace Analytics", 
            "GET", 
            "marketplace/analytics", 
            200
        )
        
        if success and isinstance(response, dict):
            print(f"   Checking marketplace analytics response structure...")
            
            # Check for expected fields
            expected_fields = ['app_statistics', 'partner_statistics', 'engagement_metrics', 'category_distribution']
            for field in expected_fields:
                if field in response:
                    print(f"   ✅ Found expected field: {field}")
                else:
                    print(f"   ⚠️  Missing expected field: {field}")
            
            # Check analytics data
            if 'app_statistics' in response:
                app_stats = response['app_statistics']
                print(f"   App statistics: {app_stats}")
            
            if 'engagement_metrics' in response:
                engagement = response['engagement_metrics']
                print(f"   Engagement metrics: {engagement}")
        
        return success, response

    # PROFESSIONAL API ENDPOINTS TESTS (6 endpoints)
    def test_developer_resources(self):
        """Test GET /api/developer-resources - Developer documentation"""
        success, response = self.run_test(
            "Developer Resources", 
            "GET", 
            "developer-resources", 
            200
        )
        
        if success and isinstance(response, dict):
            print(f"   Checking developer resources response structure...")
            
            # Check for expected fields
            expected_fields = ['api_documentation', 'sdk_libraries', 'webhook_guides', 'development_tools', 'app_submission_process']
            for field in expected_fields:
                if field in response:
                    print(f"   ✅ Found expected field: {field}")
                else:
                    print(f"   ⚠️  Missing expected field: {field}")
            
            # Check documentation structure
            if 'api_documentation' in response:
                api_docs = response['api_documentation']
                print(f"   API documentation sections: {len(api_docs) if isinstance(api_docs, (list, dict)) else 'N/A'}")
            
            if 'sdk_libraries' in response:
                sdks = response['sdk_libraries']
                print(f"   Available SDK libraries: {sdks}")
        
        return success, response

    def test_integrations_legal_research(self):
        """Test POST /api/integrations/legal-research - Professional legal research"""
        # This endpoint uses query parameters, not request body
        endpoint = "integrations/legal-research?query=breach of contract remedies&jurisdiction=US&analysis_depth=comprehensive&include_citations=true"
        
        success, response = self.run_test(
            "Professional Legal Research", 
            "POST", 
            endpoint, 
            200,
            timeout=60  # Legal research might take longer
        )
        
        if success and isinstance(response, dict):
            print(f"   Checking legal research response structure...")
            
            # Check for expected fields
            expected_fields = ['research_id', 'query', 'results', 'analysis', 'accuracy_guarantee', 'sources_checked']
            for field in expected_fields:
                if field in response:
                    print(f"   ✅ Found expected field: {field}")
                else:
                    print(f"   ⚠️  Missing expected field: {field}")
            
            # Check research results
            if 'results' in response and isinstance(response['results'], list):
                results = response['results']
                print(f"   Found {len(results)} research results")
            
            if 'accuracy_guarantee' in response:
                accuracy = response['accuracy_guarantee']
                print(f"   Accuracy guarantee: {accuracy}%")
        
        return success, response

    def test_integrations_contract_analysis(self):
        """Test POST /api/integrations/contract-analysis - Enterprise contract analysis"""
        contract_content = """
        This Service Agreement is entered into between TechCorp Inc. and ServiceProvider LLC.
        The service provider agrees to provide software development services for a period of 12 months.
        Payment terms: $10,000 per month, due within 30 days of invoice.
        Either party may terminate this agreement with 30 days written notice.
        This agreement shall be governed by the laws of California.
        """
        
        # This endpoint uses query parameters
        import urllib.parse
        encoded_content = urllib.parse.quote(contract_content.strip())
        endpoint = f"integrations/contract-analysis?contract_content={encoded_content}&contract_type=service_agreement&jurisdiction=US&analysis_level=comprehensive&include_recommendations=true"
        
        success, response = self.run_test(
            "Enterprise Contract Analysis", 
            "POST", 
            endpoint, 
            200,
            timeout=60
        )
        
        if success and isinstance(response, dict):
            print(f"   Checking contract analysis response structure...")
            
            # Check for expected fields
            expected_fields = ['analysis_id', 'risk_assessment', 'compliance_issues', 'recommendations', 'completeness_score']
            for field in expected_fields:
                if field in response:
                    print(f"   ✅ Found expected field: {field}")
                else:
                    print(f"   ⚠️  Missing expected field: {field}")
            
            # Check analysis results
            if 'risk_assessment' in response:
                risk = response['risk_assessment']
                if isinstance(risk, dict) and 'risk_score' in risk:
                    print(f"   Risk score: {risk['risk_score']}")
            
            if 'completeness_score' in response:
                completeness = response['completeness_score']
                print(f"   Completeness score: {completeness}%")
        
        return success, response

    def test_integrations_legal_memoranda(self):
        """Test POST /api/integrations/legal-memoranda - Legal memo generation"""
        # This endpoint uses query parameters
        import urllib.parse
        legal_issue = "Contract termination rights and obligations"
        facts = "Client wants to terminate a 3-year service agreement after 18 months due to vendor performance issues. Contract has a 30-day notice clause but no specific performance standards."
        
        encoded_issue = urllib.parse.quote(legal_issue)
        encoded_facts = urllib.parse.quote(facts)
        endpoint = f"integrations/legal-memoranda?legal_issue={encoded_issue}&facts={encoded_facts}&jurisdiction=US&memo_type=research&citation_style=bluebook&length=standard"
        
        success, response = self.run_test(
            "Legal Memoranda Generation", 
            "POST", 
            endpoint, 
            200,
            timeout=90  # Memo generation might take longer
        )
        
        if success and isinstance(response, dict):
            print(f"   Checking legal memoranda response structure...")
            
            # Check for expected fields
            expected_fields = ['memo_id', 'title', 'executive_summary', 'legal_analysis', 'conclusions', 'citations']
            for field in expected_fields:
                if field in response:
                    print(f"   ✅ Found expected field: {field}")
                else:
                    print(f"   ⚠️  Missing expected field: {field}")
            
            # Check memo content
            if 'legal_analysis' in response:
                analysis = response['legal_analysis']
                print(f"   Legal analysis length: {len(str(analysis))} characters")
            
            if 'citations' in response and isinstance(response['citations'], list):
                citations = response['citations']
                print(f"   Number of citations: {len(citations)}")
        
        return success, response

    def test_integrations_law_firm_dashboard(self):
        """Test GET /api/integrations/law-firm-dashboard - Law firm analytics"""
        # Test with query parameters
        endpoint = "integrations/law-firm-dashboard?law_firm_id=test-firm-123&include_benchmarks=true"
        
        success, response = self.run_test(
            "Law Firm Dashboard Analytics", 
            "GET", 
            endpoint, 
            200
        )
        
        if success and isinstance(response, dict):
            print(f"   Checking law firm dashboard response structure...")
            
            # Check for expected fields
            expected_fields = ['firm_id', 'performance_metrics', 'case_statistics', 'revenue_analytics', 'operational_insights']
            for field in expected_fields:
                if field in response:
                    print(f"   ✅ Found expected field: {field}")
                else:
                    print(f"   ⚠️  Missing expected field: {field}")
            
            # Check analytics data
            if 'performance_metrics' in response:
                performance = response['performance_metrics']
                print(f"   Performance metrics: {performance}")
            
            if 'case_statistics' in response:
                cases = response['case_statistics']
                print(f"   Case statistics: {cases}")
        
        return success, response

    def test_integrations_client_communication(self):
        """Test POST /api/integrations/client-communication - Client communication"""
        # This endpoint uses query parameters
        import urllib.parse
        client_query = "What are the main risks in this partnership agreement and should we proceed?"
        encoded_query = urllib.parse.quote(client_query)
        endpoint = f"integrations/client-communication?client_query={encoded_query}&communication_type=email&tone=professional&include_disclaimers=true"
        
        success, response = self.run_test(
            "Client Communication Generation", 
            "POST", 
            endpoint, 
            200,
            timeout=60
        )
        
        if success and isinstance(response, dict):
            print(f"   Checking client communication response structure...")
            
            # Check for expected fields
            expected_fields = ['communication_id', 'subject', 'content', 'legal_accuracy_score', 'compliance_notes', 'review_required']
            for field in expected_fields:
                if field in response:
                    print(f"   ✅ Found expected field: {field}")
                else:
                    print(f"   ⚠️  Missing expected field: {field}")
            
            # Check communication content
            if 'content' in response:
                content = response['content']
                print(f"   Communication content length: {len(str(content))} characters")
            
            if 'legal_accuracy_score' in response:
                accuracy = response['legal_accuracy_score']
                print(f"   Legal accuracy score: {accuracy}%")
        
        return success, response

    def test_integrations_billing_optimization(self):
        """Test GET /api/integrations/billing-optimization - Billing optimization"""
        # Test with query parameters
        endpoint = "integrations/billing-optimization?law_firm_id=test-firm-123&analysis_period=quarterly&include_recommendations=true"
        
        success, response = self.run_test(
            "Billing Optimization Analytics", 
            "GET", 
            endpoint, 
            200
        )
        
        if success and isinstance(response, dict):
            print(f"   Checking billing optimization response structure...")
            
            # Check for expected fields
            expected_fields = ['firm_id', 'billing_analytics', 'optimization_opportunities', 'revenue_projections', 'benchmarks']
            for field in expected_fields:
                if field in response:
                    print(f"   ✅ Found expected field: {field}")
                else:
                    print(f"   ⚠️  Missing expected field: {field}")
            
            # Check optimization data
            if 'optimization_opportunities' in response and isinstance(response['optimization_opportunities'], list):
                opportunities = response['optimization_opportunities']
                print(f"   Found {len(opportunities)} optimization opportunities")
            
            if 'revenue_projections' in response:
                projections = response['revenue_projections']
                print(f"   Revenue projections: {projections}")
        
        return success, response

    def run_all_tests(self):
        """Run all professional integrations tests"""
        print("🚀 Starting Professional Integrations & API Ecosystem Testing")
        print("=" * 80)
        
        # Professional Integrations Framework Tests
        print("\n📋 PROFESSIONAL INTEGRATIONS FRAMEWORK TESTS")
        print("-" * 50)
        self.test_integrations_status()
        self.test_integrations_activate()
        self.test_integrations_action()
        
        # Professional API Ecosystem Tests
        print("\n🔑 PROFESSIONAL API ECOSYSTEM TESTS")
        print("-" * 50)
        self.test_api_ecosystem_generate_key()
        self.test_api_ecosystem_documentation()
        self.test_api_ecosystem_usage_analytics()
        
        # Enterprise Integration Features Tests
        print("\n🏢 ENTERPRISE INTEGRATION FEATURES TESTS")
        print("-" * 50)
        self.test_enterprise_sso_authenticate()
        self.test_enterprise_compliance_check()
        self.test_enterprise_audit_trail()
        
        # Legal Workflow Automation Tests
        print("\n⚙️ LEGAL WORKFLOW AUTOMATION TESTS")
        print("-" * 50)
        self.test_workflows_templates()
        self.test_workflows_create()
        self.test_workflows_start()
        self.test_workflows_status()
        self.test_workflows_analytics()
        
        # Marketplace & Partnership Ecosystem Tests
        print("\n🏪 MARKETPLACE & PARTNERSHIP ECOSYSTEM TESTS")
        print("-" * 50)
        self.test_marketplace_search()
        self.test_marketplace_app_details()
        self.test_marketplace_install()
        self.test_marketplace_review()
        self.test_partnerships_apply()
        self.test_partnerships_search()
        self.test_marketplace_analytics()
        
        # Professional API Endpoints Tests
        print("\n🔧 PROFESSIONAL API ENDPOINTS TESTS")
        print("-" * 50)
        self.test_developer_resources()
        self.test_integrations_legal_research()
        self.test_integrations_contract_analysis()
        self.test_integrations_legal_memoranda()
        self.test_integrations_law_firm_dashboard()
        self.test_integrations_client_communication()
        self.test_integrations_billing_optimization()
        
        # Final Results
        print("\n" + "=" * 80)
        print("🎯 PROFESSIONAL INTEGRATIONS TESTING COMPLETE")
        print("=" * 80)
        print(f"Total Tests Run: {self.tests_run}")
        print(f"Tests Passed: {self.tests_passed}")
        print(f"Tests Failed: {self.tests_run - self.tests_passed}")
        print(f"Success Rate: {(self.tests_passed / self.tests_run * 100):.1f}%")
        
        if self.tests_passed == self.tests_run:
            print("🎉 ALL TESTS PASSED! Professional Integrations & API Ecosystem is fully operational.")
        elif self.tests_passed / self.tests_run >= 0.8:
            print("✅ MOSTLY SUCCESSFUL! Professional Integrations system is largely operational with minor issues.")
        else:
            print("⚠️ SIGNIFICANT ISSUES DETECTED! Professional Integrations system needs attention.")
        
        return self.tests_passed, self.tests_run

if __name__ == "__main__":
    tester = ProfessionalIntegrationsAPITester()
    passed, total = tester.run_all_tests()
    
    # Exit with appropriate code
    if passed == total:
        sys.exit(0)  # All tests passed
    else:
        sys.exit(1)  # Some tests failed