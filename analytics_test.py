import requests
import sys
import json
from datetime import datetime, timedelta

class AnalyticsAPITester:
    def __init__(self, base_url="https://legalengine.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.failed_tests = []

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
                self.failed_tests.append(name)
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data}")
                except:
                    print(f"   Error: {response.text}")
                return False, {}

        except requests.exceptions.Timeout:
            print(f"❌ Failed - Request timed out after {timeout} seconds")
            self.failed_tests.append(name)
            return False, {}
        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            self.failed_tests.append(name)
            return False, {}

    # BASIC ANALYTICS ENDPOINTS TESTS
    def test_analytics_dashboard(self):
        """Test GET /api/analytics/dashboard - Dashboard overview data with filtering capabilities"""
        # Test without filters
        success, response = self.run_test(
            "Analytics Dashboard (No Filters)", 
            "GET", 
            "analytics/dashboard", 
            200
        )
        
        if success:
            # Verify response structure
            expected_keys = ['overview', 'contract_distribution', 'trends', 'filters_applied']
            for key in expected_keys:
                if key in response:
                    print(f"   ✅ Found expected key: {key}")
                else:
                    print(f"   ⚠️  Missing expected key: {key}")
            
            # Check overview data
            if 'overview' in response:
                overview = response['overview']
                print(f"   Overview data: {overview}")
        
        # Test with date range filter
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        success_filtered, response_filtered = self.run_test(
            "Analytics Dashboard (Date Filter)", 
            "GET", 
            f"analytics/dashboard?start_date={start_date}&end_date={end_date}", 
            200
        )
        
        if success_filtered and 'filters_applied' in response_filtered:
            filters = response_filtered['filters_applied']
            print(f"   Applied filters: {filters}")
        
        # Test with contract type filter
        success_type, response_type = self.run_test(
            "Analytics Dashboard (Contract Type Filter)", 
            "GET", 
            "analytics/dashboard?contract_type=NDA", 
            200
        )
        
        # Test with jurisdiction filter
        success_jurisdiction, response_jurisdiction = self.run_test(
            "Analytics Dashboard (Jurisdiction Filter)", 
            "GET", 
            "analytics/dashboard?jurisdiction=US", 
            200
        )
        
        return success and success_filtered and success_type and success_jurisdiction, response

    def test_performance_metrics(self):
        """Test GET /api/analytics/performance-metrics - Contract performance metrics"""
        success, response = self.run_test(
            "Performance Metrics", 
            "GET", 
            "analytics/performance-metrics", 
            200
        )
        
        if success:
            # Verify expected metrics
            expected_metrics = [
                'total_contracts', 'success_rate', 'average_compliance_score',
                'dispute_frequency', 'renewal_rate', 'client_satisfaction',
                'time_to_completion_avg', 'cost_savings_total', 'efficiency_improvement'
            ]
            
            for metric in expected_metrics:
                if metric in response:
                    value = response[metric]
                    print(f"   ✅ {metric}: {value}")
                    
                    # Validate metric ranges
                    if metric in ['success_rate', 'average_compliance_score', 'client_satisfaction']:
                        if isinstance(value, (int, float)) and 0 <= value <= 100:
                            print(f"     ✅ Valid percentage range (0-100)")
                        else:
                            print(f"     ⚠️  Invalid percentage value: {value}")
                    
                    elif metric == 'dispute_frequency':
                        if isinstance(value, (int, float)) and value >= 0:
                            print(f"     ✅ Valid dispute frequency")
                        else:
                            print(f"     ⚠️  Invalid dispute frequency: {value}")
                else:
                    print(f"   ⚠️  Missing expected metric: {metric}")
        
        return success, response

    def test_cost_analysis(self):
        """Test GET /api/analytics/cost-analysis - Cost analysis and savings calculations"""
        success, response = self.run_test(
            "Cost Analysis", 
            "GET", 
            "analytics/cost-analysis", 
            200
        )
        
        if success:
            # Verify cost analysis structure
            expected_keys = [
                'total_savings', 'time_saved_hours', 'cost_per_contract',
                'savings_percentage', 'roi', 'process_breakdown'
            ]
            
            for key in expected_keys:
                if key in response:
                    value = response[key]
                    print(f"   ✅ {key}: {value}")
                    
                    # Validate specific metrics
                    if key == 'savings_percentage':
                        if isinstance(value, (int, float)) and 0 <= value <= 100:
                            print(f"     ✅ Valid savings percentage")
                        else:
                            print(f"     ⚠️  Invalid savings percentage: {value}")
                    
                    elif key == 'roi':
                        if isinstance(value, (int, float)) and value >= 0:
                            print(f"     ✅ Valid ROI value")
                        else:
                            print(f"     ⚠️  Invalid ROI value: {value}")
                    
                    elif key == 'process_breakdown':
                        if isinstance(value, dict):
                            print(f"     ✅ Process breakdown is a dictionary")
                            for process, cost in value.items():
                                print(f"       - {process}: {cost}")
                        else:
                            print(f"     ⚠️  Process breakdown should be a dictionary")
                else:
                    print(f"   ⚠️  Missing expected key: {key}")
        
        return success, response

    def test_negotiation_insights(self):
        """Test GET /api/analytics/negotiation-insights - Negotiation patterns and insights"""
        success, response = self.run_test(
            "Negotiation Insights", 
            "GET", 
            "analytics/negotiation-insights", 
            200
        )
        
        if success:
            # Verify negotiation insights structure
            expected_keys = [
                'total_negotiations', 'average_rounds', 'success_rate',
                'common_negotiation_points', 'effective_strategies', 'seasonal_trends'
            ]
            
            for key in expected_keys:
                if key in response:
                    value = response[key]
                    print(f"   ✅ {key}: {value}")
                    
                    # Validate specific metrics
                    if key == 'success_rate':
                        if isinstance(value, (int, float)) and 0 <= value <= 100:
                            print(f"     ✅ Valid success rate")
                        else:
                            print(f"     ⚠️  Invalid success rate: {value}")
                    
                    elif key in ['common_negotiation_points', 'effective_strategies']:
                        if isinstance(value, list):
                            print(f"     ✅ {key} is a list with {len(value)} items")
                        else:
                            print(f"     ⚠️  {key} should be a list")
                    
                    elif key == 'seasonal_trends':
                        if isinstance(value, dict):
                            print(f"     ✅ Seasonal trends is a dictionary")
                        else:
                            print(f"     ⚠️  Seasonal trends should be a dictionary")
                else:
                    print(f"   ⚠️  Missing expected key: {key}")
        
        return success, response

    def test_market_intelligence(self):
        """Test GET /api/analytics/market-intelligence - AI-powered market intelligence"""
        # Test without parameters
        success, response = self.run_test(
            "Market Intelligence (No Parameters)", 
            "GET", 
            "analytics/market-intelligence", 
            200
        )
        
        if success:
            # Verify market intelligence structure
            expected_keys = [
                'ai_analysis', 'industry_benchmarks', 'market_trends',
                'competitive_analysis', 'recommendations'
            ]
            
            for key in expected_keys:
                if key in response:
                    value = response[key]
                    print(f"   ✅ {key}: {type(value).__name__}")
                    
                    if key == 'ai_analysis':
                        if isinstance(value, str) and len(value) > 100:
                            print(f"     ✅ AI analysis has substantial content ({len(value)} characters)")
                        else:
                            print(f"     ⚠️  AI analysis seems too short: {len(value) if isinstance(value, str) else 'Not a string'}")
                    
                    elif key in ['market_trends', 'recommendations']:
                        if isinstance(value, list):
                            print(f"     ✅ {key} is a list with {len(value)} items")
                        else:
                            print(f"     ⚠️  {key} should be a list")
                else:
                    print(f"   ⚠️  Missing expected key: {key}")
        
        # Test with industry parameter
        success_industry, response_industry = self.run_test(
            "Market Intelligence (Industry Filter)", 
            "GET", 
            "analytics/market-intelligence?industry=technology", 
            200
        )
        
        # Test with contract type parameter
        success_contract, response_contract = self.run_test(
            "Market Intelligence (Contract Type Filter)", 
            "GET", 
            "analytics/market-intelligence?contract_type=NDA", 
            200
        )
        
        # Test with jurisdiction parameter
        success_jurisdiction, response_jurisdiction = self.run_test(
            "Market Intelligence (Jurisdiction Filter)", 
            "GET", 
            "analytics/market-intelligence?jurisdiction=US", 
            200
        )
        
        return success and success_industry and success_contract and success_jurisdiction, response

    def test_track_event(self):
        """Test POST /api/analytics/track-event - Event tracking functionality"""
        # Test negotiation event
        negotiation_event = {
            "event_type": "negotiation",
            "contract_id": "test-contract-123",
            "event_data": {
                "negotiation_round": 2,
                "outcome": "successful",
                "duration_minutes": 45,
                "key_points": ["payment terms", "delivery schedule"]
            },
            "user_id": "test-user-456"
        }
        
        success_negotiation, response_negotiation = self.run_test(
            "Track Event (Negotiation)", 
            "POST", 
            "analytics/track-event", 
            200,
            negotiation_event
        )
        
        if success_negotiation:
            if 'event_id' in response_negotiation:
                print(f"   ✅ Event tracked with ID: {response_negotiation['event_id']}")
            else:
                print(f"   ⚠️  No event_id returned")
        
        # Test dispute event
        dispute_event = {
            "event_type": "dispute",
            "contract_id": "test-contract-789",
            "event_data": {
                "dispute_type": "payment",
                "severity": "medium",
                "resolution_status": "pending",
                "description": "Payment terms disagreement"
            },
            "user_id": "test-user-789"
        }
        
        success_dispute, response_dispute = self.run_test(
            "Track Event (Dispute)", 
            "POST", 
            "analytics/track-event", 
            200,
            dispute_event
        )
        
        # Test renewal event
        renewal_event = {
            "event_type": "renewal",
            "contract_id": "test-contract-456",
            "event_data": {
                "renewal_type": "automatic",
                "new_term_length": "12_months",
                "rate_change": 5.0,
                "renewal_date": "2025-06-01"
            },
            "user_id": "test-user-123"
        }
        
        success_renewal, response_renewal = self.run_test(
            "Track Event (Renewal)", 
            "POST", 
            "analytics/track-event", 
            200,
            renewal_event
        )
        
        return success_negotiation and success_dispute and success_renewal, {
            "negotiation": response_negotiation,
            "dispute": response_dispute,
            "renewal": response_renewal
        }

    # ADVANCED ANALYTICS ENDPOINTS TESTS
    def test_advanced_metrics(self):
        """Test GET /api/analytics/advanced-metrics - Advanced growth metrics"""
        success, response = self.run_test(
            "Advanced Metrics", 
            "GET", 
            "analytics/advanced-metrics", 
            200
        )
        
        if success:
            # Verify advanced metrics structure
            expected_keys = [
                'growth_metrics', 'trend_analysis', 'forecasting',
                'user_engagement', 'contract_lifecycle', 'performance_indicators'
            ]
            
            for key in expected_keys:
                if key in response:
                    value = response[key]
                    print(f"   ✅ {key}: {type(value).__name__}")
                    
                    if key == 'growth_metrics':
                        if isinstance(value, dict):
                            print(f"     ✅ Growth metrics is a dictionary")
                            for metric, val in value.items():
                                print(f"       - {metric}: {val}")
                        else:
                            print(f"     ⚠️  Growth metrics should be a dictionary")
                    
                    elif key == 'forecasting':
                        if isinstance(value, dict):
                            print(f"     ✅ Forecasting data is a dictionary")
                        else:
                            print(f"     ⚠️  Forecasting should be a dictionary")
                else:
                    print(f"   ⚠️  Missing expected key: {key}")
        
        return success, response

    def test_real_time_stats(self):
        """Test GET /api/analytics/real-time-stats - Real-time statistics"""
        success, response = self.run_test(
            "Real-time Stats", 
            "GET", 
            "analytics/real-time-stats", 
            200
        )
        
        if success:
            # Verify real-time stats structure
            expected_keys = [
                'active_users', 'contracts_in_progress', 'recent_activities',
                'system_health', 'api_usage', 'live_metrics'
            ]
            
            for key in expected_keys:
                if key in response:
                    value = response[key]
                    print(f"   ✅ {key}: {value}")
                    
                    if key == 'recent_activities':
                        if isinstance(value, list):
                            print(f"     ✅ Recent activities is a list with {len(value)} items")
                        else:
                            print(f"     ⚠️  Recent activities should be a list")
                    
                    elif key == 'system_health':
                        if isinstance(value, dict):
                            print(f"     ✅ System health is a dictionary")
                            for component, status in value.items():
                                print(f"       - {component}: {status}")
                        else:
                            print(f"     ⚠️  System health should be a dictionary")
                else:
                    print(f"   ⚠️  Missing expected key: {key}")
        
        return success, response

    def test_compliance_deep_dive(self):
        """Test GET /api/analytics/compliance-deep-dive - Compliance analysis"""
        success, response = self.run_test(
            "Compliance Deep Dive", 
            "GET", 
            "analytics/compliance-deep-dive", 
            200
        )
        
        if success:
            # Verify compliance deep dive structure
            expected_keys = [
                'overall_compliance_score', 'jurisdiction_breakdown', 'compliance_trends',
                'risk_assessment', 'regulatory_updates', 'improvement_recommendations'
            ]
            
            for key in expected_keys:
                if key in response:
                    value = response[key]
                    print(f"   ✅ {key}: {type(value).__name__}")
                    
                    if key == 'overall_compliance_score':
                        if isinstance(value, (int, float)) and 0 <= value <= 100:
                            print(f"     ✅ Valid compliance score: {value}")
                        else:
                            print(f"     ⚠️  Invalid compliance score: {value}")
                    
                    elif key == 'jurisdiction_breakdown':
                        if isinstance(value, dict):
                            print(f"     ✅ Jurisdiction breakdown is a dictionary")
                            for jurisdiction, score in value.items():
                                print(f"       - {jurisdiction}: {score}")
                        else:
                            print(f"     ⚠️  Jurisdiction breakdown should be a dictionary")
                    
                    elif key in ['regulatory_updates', 'improvement_recommendations']:
                        if isinstance(value, list):
                            print(f"     ✅ {key} is a list with {len(value)} items")
                        else:
                            print(f"     ⚠️  {key} should be a list")
                else:
                    print(f"   ⚠️  Missing expected key: {key}")
        
        return success, response

    def test_integration_metrics(self):
        """Test GET /api/analytics/integration-metrics - System and API metrics"""
        success, response = self.run_test(
            "Integration Metrics", 
            "GET", 
            "analytics/integration-metrics", 
            200
        )
        
        if success:
            # Verify integration metrics structure
            expected_keys = [
                'api_performance', 'system_uptime', 'integration_health',
                'third_party_services', 'error_rates', 'response_times'
            ]
            
            for key in expected_keys:
                if key in response:
                    value = response[key]
                    print(f"   ✅ {key}: {type(value).__name__}")
                    
                    if key == 'api_performance':
                        if isinstance(value, dict):
                            print(f"     ✅ API performance is a dictionary")
                            for endpoint, metrics in value.items():
                                print(f"       - {endpoint}: {metrics}")
                        else:
                            print(f"     ⚠️  API performance should be a dictionary")
                    
                    elif key == 'system_uptime':
                        if isinstance(value, (int, float)) and 0 <= value <= 100:
                            print(f"     ✅ Valid system uptime: {value}%")
                        else:
                            print(f"     ⚠️  Invalid system uptime: {value}")
                    
                    elif key == 'third_party_services':
                        if isinstance(value, dict):
                            print(f"     ✅ Third party services is a dictionary")
                            for service, status in value.items():
                                print(f"       - {service}: {status}")
                        else:
                            print(f"     ⚠️  Third party services should be a dictionary")
                else:
                    print(f"   ⚠️  Missing expected key: {key}")
        
        return success, response

    def test_export_data(self):
        """Test POST /api/analytics/export-data - Data export functionality"""
        # Test CSV export
        csv_export_request = {
            "export_format": "csv",
            "data_type": "contracts",
            "date_range": {
                "start_date": "2024-01-01",
                "end_date": "2025-01-31"
            },
            "filters": {
                "contract_type": "NDA",
                "jurisdiction": "US"
            }
        }
        
        success_csv, response_csv = self.run_test(
            "Export Data (CSV)", 
            "POST", 
            "analytics/export-data", 
            200,
            csv_export_request
        )
        
        if success_csv:
            if 'export_id' in response_csv:
                print(f"   ✅ CSV export initiated with ID: {response_csv['export_id']}")
            if 'download_url' in response_csv:
                print(f"   ✅ Download URL provided: {response_csv['download_url']}")
        
        # Test JSON export
        json_export_request = {
            "export_format": "json",
            "data_type": "analytics",
            "date_range": {
                "start_date": "2024-12-01",
                "end_date": "2025-01-31"
            },
            "include_metadata": True
        }
        
        success_json, response_json = self.run_test(
            "Export Data (JSON)", 
            "POST", 
            "analytics/export-data", 
            200,
            json_export_request
        )
        
        # Test Excel export
        excel_export_request = {
            "export_format": "excel",
            "data_type": "performance_metrics",
            "date_range": {
                "start_date": "2024-01-01",
                "end_date": "2025-01-31"
            },
            "include_charts": True
        }
        
        success_excel, response_excel = self.run_test(
            "Export Data (Excel)", 
            "POST", 
            "analytics/export-data", 
            200,
            excel_export_request
        )
        
        return success_csv and success_json and success_excel, {
            "csv": response_csv,
            "json": response_json,
            "excel": response_excel
        }

    def test_predictive_insights(self):
        """Test GET /api/analytics/predictive-insights - AI-powered predictions"""
        success, response = self.run_test(
            "Predictive Insights", 
            "GET", 
            "analytics/predictive-insights", 
            200
        )
        
        if success:
            # Verify predictive insights structure
            expected_keys = [
                'contract_success_predictions', 'market_forecasts', 'risk_predictions',
                'trend_predictions', 'recommendations', 'confidence_scores'
            ]
            
            for key in expected_keys:
                if key in response:
                    value = response[key]
                    print(f"   ✅ {key}: {type(value).__name__}")
                    
                    if key == 'contract_success_predictions':
                        if isinstance(value, list):
                            print(f"     ✅ Contract success predictions is a list with {len(value)} items")
                            for prediction in value[:3]:  # Show first 3
                                if isinstance(prediction, dict):
                                    print(f"       - Contract: {prediction.get('contract_id', 'N/A')}, Success Probability: {prediction.get('success_probability', 'N/A')}")
                        else:
                            print(f"     ⚠️  Contract success predictions should be a list")
                    
                    elif key == 'confidence_scores':
                        if isinstance(value, dict):
                            print(f"     ✅ Confidence scores is a dictionary")
                            for prediction_type, score in value.items():
                                if isinstance(score, (int, float)) and 0 <= score <= 1:
                                    print(f"       - {prediction_type}: {score:.2f}")
                                else:
                                    print(f"       - {prediction_type}: {score} (⚠️  Invalid confidence score)")
                        else:
                            print(f"     ⚠️  Confidence scores should be a dictionary")
                    
                    elif key in ['market_forecasts', 'risk_predictions', 'trend_predictions', 'recommendations']:
                        if isinstance(value, list):
                            print(f"     ✅ {key} is a list with {len(value)} items")
                        else:
                            print(f"     ⚠️  {key} should be a list")
                else:
                    print(f"   ⚠️  Missing expected key: {key}")
        
        return success, response

    def run_all_analytics_tests(self):
        """Run all analytics endpoint tests"""
        print("🚀 Starting Comprehensive Analytics Backend Testing")
        print("=" * 60)
        
        # Basic Analytics Endpoints
        print("\n📊 BASIC ANALYTICS ENDPOINTS")
        print("-" * 40)
        
        self.test_analytics_dashboard()
        self.test_performance_metrics()
        self.test_cost_analysis()
        self.test_negotiation_insights()
        self.test_market_intelligence()
        self.test_track_event()
        
        # Advanced Analytics Endpoints
        print("\n🔬 ADVANCED ANALYTICS ENDPOINTS")
        print("-" * 40)
        
        self.test_advanced_metrics()
        self.test_real_time_stats()
        self.test_compliance_deep_dive()
        self.test_integration_metrics()
        self.test_export_data()
        self.test_predictive_insights()
        
        # Summary
        print("\n" + "=" * 60)
        print("📋 ANALYTICS TESTING SUMMARY")
        print("=" * 60)
        print(f"Total Tests Run: {self.tests_run}")
        print(f"Tests Passed: {self.tests_passed}")
        print(f"Tests Failed: {self.tests_run - self.tests_passed}")
        print(f"Success Rate: {(self.tests_passed / self.tests_run * 100):.1f}%")
        
        if self.failed_tests:
            print(f"\n❌ Failed Tests:")
            for test in self.failed_tests:
                print(f"   - {test}")
        else:
            print(f"\n🎉 All analytics endpoints are working perfectly!")
        
        return self.tests_passed == self.tests_run

if __name__ == "__main__":
    tester = AnalyticsAPITester()
    success = tester.run_all_analytics_tests()
    
    if success:
        print(f"\n✅ All analytics endpoints verified successfully!")
        sys.exit(0)
    else:
        print(f"\n❌ Some analytics endpoints failed verification.")
        sys.exit(1)