import requests
import sys
import json
from datetime import datetime, timedelta

class AnalyticsFixedTester:
    def __init__(self, base_url="https://legal-research-api.preview.emergentagent.com"):
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

    def test_export_data_fixed(self):
        """Test POST /api/analytics/export-data with correct parameters"""
        # Test CSV export with correct structure
        csv_export_request = {
            "export_type": "csv",  # Changed from export_format
            "data_types": ["overview", "performance"],  # Changed from data_type to data_types (list)
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
            "Export Data (CSV - Fixed)", 
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
        
        # Test JSON export with correct structure
        json_export_request = {
            "export_type": "json",  # Changed from export_format
            "data_types": ["analytics", "costs"],  # Changed from data_type to data_types (list)
            "date_range": {
                "start_date": "2024-12-01",
                "end_date": "2025-01-31"
            }
        }
        
        success_json, response_json = self.run_test(
            "Export Data (JSON - Fixed)", 
            "POST", 
            "analytics/export-data", 
            200,
            json_export_request
        )
        
        # Test Excel export with correct structure
        excel_export_request = {
            "export_type": "excel",  # Changed from export_format
            "data_types": ["performance", "negotiations"],  # Changed from data_type to data_types (list)
            "date_range": {
                "start_date": "2024-01-01",
                "end_date": "2025-01-31"
            }
        }
        
        success_excel, response_excel = self.run_test(
            "Export Data (Excel - Fixed)", 
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

    def test_predictive_insights_fixed(self):
        """Test GET /api/analytics/predictive-insights with required parameters"""
        # Test with required contract_type parameter
        success, response = self.run_test(
            "Predictive Insights (Required Params)", 
            "GET", 
            "analytics/predictive-insights?contract_type=NDA&jurisdiction=US", 
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
                else:
                    print(f"   ⚠️  Missing expected key: {key}")
        
        # Test with additional parameters
        success_full, response_full = self.run_test(
            "Predictive Insights (Full Params)", 
            "GET", 
            "analytics/predictive-insights?contract_type=freelance_agreement&jurisdiction=US&industry=technology&party_count=2", 
            200
        )
        
        return success and success_full, response

    def test_market_intelligence_timeout_fix(self):
        """Test market intelligence with shorter timeout to avoid hanging"""
        success, response = self.run_test(
            "Market Intelligence (Industry Filter - Short Timeout)", 
            "GET", 
            "analytics/market-intelligence?industry=technology", 
            200,
            timeout=60  # Increased timeout for AI processing
        )
        
        if success:
            print(f"   ✅ Market intelligence with industry filter working")
            if 'ai_generated_insights' in response:
                insights = response['ai_generated_insights']
                if isinstance(insights, str) and len(insights) > 100:
                    print(f"     ✅ AI insights generated ({len(insights)} characters)")
                else:
                    print(f"     ⚠️  AI insights seem limited")
        
        return success, response

    def run_fixed_tests(self):
        """Run the fixed tests for failing endpoints"""
        print("🔧 Running Fixed Tests for Previously Failed Endpoints")
        print("=" * 60)
        
        # Test the fixed export data endpoint
        print("\n📤 EXPORT DATA ENDPOINT (FIXED)")
        print("-" * 40)
        self.test_export_data_fixed()
        
        # Test the fixed predictive insights endpoint
        print("\n🔮 PREDICTIVE INSIGHTS ENDPOINT (FIXED)")
        print("-" * 40)
        self.test_predictive_insights_fixed()
        
        # Test market intelligence with longer timeout
        print("\n🧠 MARKET INTELLIGENCE TIMEOUT FIX")
        print("-" * 40)
        self.test_market_intelligence_timeout_fix()
        
        # Summary
        print("\n" + "=" * 60)
        print("📋 FIXED TESTS SUMMARY")
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
            print(f"\n🎉 All fixed tests are working!")
        
        return self.tests_passed == self.tests_run

if __name__ == "__main__":
    tester = AnalyticsFixedTester()
    success = tester.run_fixed_tests()
    
    if success:
        print(f"\n✅ All fixed analytics endpoints verified successfully!")
        sys.exit(0)
    else:
        print(f"\n❌ Some fixed analytics endpoints still have issues.")
        sys.exit(1)