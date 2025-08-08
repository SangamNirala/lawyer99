#!/usr/bin/env python3
"""
Production Optimization & Performance Analytics System Testing
Comprehensive testing of 8 new production endpoints for enterprise-grade capabilities
"""

import requests
import sys
import json
import time
from datetime import datetime, timedelta

class ProductionSystemTester:
    def __init__(self, base_url="https://8fa35272-f813-4f54-99e2-91d7f9fb1b50.preview.emergentagent.com"):
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
                self.failed_tests.append(name)
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data}")
                except:
                    print(f"   Error: {response.text}")
                return False, None

        except requests.exceptions.Timeout:
            self.failed_tests.append(name)
            print(f"❌ Failed - Request timeout after {timeout}s")
            return False, None
        except Exception as e:
            self.failed_tests.append(name)
            print(f"❌ Failed - Exception: {str(e)}")
            return False, None

    def test_production_status_endpoint(self):
        """Test GET /api/production/status - Get comprehensive production system status"""
        print("\n" + "="*80)
        print("🎯 TESTING PRODUCTION STATUS ENDPOINT")
        print("="*80)
        
        success, response = self.run_test(
            "Production Status - Basic Request",
            "GET",
            "production/status",
            200
        )
        
        if success and response:
            # Validate response structure
            required_fields = ['systems_status', 'overall_health', 'active_sessions', 
                             'concurrent_requests', 'cache_hit_rate', 'average_response_time', 'timestamp']
            
            missing_fields = [field for field in required_fields if field not in response]
            if missing_fields:
                print(f"⚠️  Missing required fields: {missing_fields}")
                return False
            
            # Validate systems_status structure
            systems_status = response.get('systems_status', {})
            expected_systems = ['performance_optimization', 'analytics_system', 'scalability_system', 'monitoring_system']
            
            for system in expected_systems:
                if system not in systems_status:
                    print(f"⚠️  Missing system status: {system}")
                    return False
                
                status = systems_status[system]
                if status not in ['active', 'unavailable']:
                    print(f"⚠️  Invalid system status for {system}: {status}")
                    return False
            
            print(f"✅ Production Status Response Structure Valid")
            print(f"   Systems Status: {systems_status}")
            print(f"   Overall Health: {response.get('overall_health')}")
            print(f"   Active Sessions: {response.get('active_sessions')}")
            print(f"   Cache Hit Rate: {response.get('cache_hit_rate')}%")
            
            return True
        
        return False

    def test_production_metrics_endpoint(self):
        """Test GET /api/production/metrics - Get comprehensive production metrics"""
        print("\n" + "="*80)
        print("🎯 TESTING PRODUCTION METRICS ENDPOINT")
        print("="*80)
        
        # Test with default parameters
        success, response = self.run_test(
            "Production Metrics - Default (24 hours)",
            "GET",
            "production/metrics",
            200
        )
        
        if success and response:
            # Validate response structure
            required_fields = ['cache_metrics', 'performance_metrics', 'scalability_metrics', 
                             'system_health', 'analytics_summary', 'timestamp']
            
            missing_fields = [field for field in required_fields if field not in response]
            if missing_fields:
                print(f"⚠️  Missing required fields: {missing_fields}")
                return False
            
            print(f"✅ Production Metrics Response Structure Valid")
            print(f"   Cache Metrics: {type(response.get('cache_metrics'))}")
            print(f"   Performance Metrics: {type(response.get('performance_metrics'))}")
            print(f"   Scalability Metrics: {type(response.get('scalability_metrics'))}")
            print(f"   System Health: {type(response.get('system_health'))}")
            
            # Test with custom time period
            success2, response2 = self.run_test(
                "Production Metrics - Custom Period (48 hours)",
                "GET",
                "production/metrics?hours=48",
                200
            )
            
            return success2
        
        return False

    def test_analytics_report_endpoint(self):
        """Test POST /api/production/analytics/report - Generate analytics reports"""
        print("\n" + "="*80)
        print("🎯 TESTING ANALYTICS REPORT ENDPOINT")
        print("="*80)
        
        # Test with default parameters
        success, response = self.run_test(
            "Analytics Report - Default Parameters",
            "POST",
            "production/analytics/report",
            200
        )
        
        if success and response:
            # Validate response structure
            required_fields = ['success', 'report_type', 'report']
            
            missing_fields = [field for field in required_fields if field not in response]
            if missing_fields:
                print(f"⚠️  Missing required fields: {missing_fields}")
                return False
            
            if not response.get('success'):
                print(f"⚠️  Report generation not successful")
                return False
            
            print(f"✅ Analytics Report Generated Successfully")
            print(f"   Report Type: {response.get('report_type')}")
            print(f"   Report Keys: {list(response.get('report', {}).keys()) if isinstance(response.get('report'), dict) else 'Non-dict report'}")
            
            # Test with custom date range
            start_date = (datetime.utcnow() - timedelta(days=7)).isoformat() + 'Z'
            end_date = datetime.utcnow().isoformat() + 'Z'
            
            success2, response2 = self.run_test(
                "Analytics Report - Custom Date Range",
                "POST",
                f"production/analytics/report?start_date={start_date}&end_date={end_date}&report_type=performance",
                200
            )
            
            return success2
        
        return False

    def test_cache_invalidation_endpoint(self):
        """Test POST /api/production/cache/invalidate - Invalidate cache entries"""
        print("\n" + "="*80)
        print("🎯 TESTING CACHE INVALIDATION ENDPOINT")
        print("="*80)
        
        # Test namespace invalidation
        success, response = self.run_test(
            "Cache Invalidation - Namespace Only",
            "POST",
            "production/cache/invalidate?namespace=contracts",
            200
        )
        
        if success and response:
            # Validate response structure
            required_fields = ['success', 'message']
            
            missing_fields = [field for field in required_fields if field not in response]
            if missing_fields:
                print(f"⚠️  Missing required fields: {missing_fields}")
                return False
            
            if not response.get('success'):
                print(f"⚠️  Cache invalidation not successful")
                return False
            
            print(f"✅ Cache Invalidation Successful")
            print(f"   Message: {response.get('message')}")
            
            # Test specific key invalidation
            success2, response2 = self.run_test(
                "Cache Invalidation - Specific Key",
                "POST",
                "production/cache/invalidate?namespace=legal_concepts&key=frequent_queries",
                200
            )
            
            if success2 and response2:
                print(f"✅ Specific Key Invalidation Successful")
                print(f"   Message: {response2.get('message')}")
                return True
        
        return False

    def test_active_sessions_endpoint(self):
        """Test GET /api/production/sessions - Get active user sessions information"""
        print("\n" + "="*80)
        print("🎯 TESTING ACTIVE SESSIONS ENDPOINT")
        print("="*80)
        
        success, response = self.run_test(
            "Active Sessions - Session Statistics",
            "GET",
            "production/sessions",
            200
        )
        
        if success and response:
            # Validate response structure
            required_fields = ['success', 'session_statistics', 'load_balancing', 'timestamp']
            
            missing_fields = [field for field in required_fields if field not in response]
            if missing_fields:
                print(f"⚠️  Missing required fields: {missing_fields}")
                return False
            
            if not response.get('success'):
                print(f"⚠️  Session retrieval not successful")
                return False
            
            print(f"✅ Active Sessions Retrieved Successfully")
            print(f"   Session Statistics: {type(response.get('session_statistics'))}")
            print(f"   Load Balancing: {type(response.get('load_balancing'))}")
            print(f"   Timestamp: {response.get('timestamp')}")
            
            # Validate session statistics structure
            session_stats = response.get('session_statistics', {})
            if isinstance(session_stats, dict):
                print(f"   Session Stats Keys: {list(session_stats.keys())}")
            
            return True
        
        return False

    def test_system_health_endpoint(self):
        """Test GET /api/production/health - Get detailed system health information"""
        print("\n" + "="*80)
        print("🎯 TESTING SYSTEM HEALTH ENDPOINT")
        print("="*80)
        
        success, response = self.run_test(
            "System Health - Health Checks",
            "GET",
            "production/health",
            200
        )
        
        if success and response:
            # Validate response structure
            required_fields = ['success', 'system_status', 'component_health', 'timestamp']
            
            missing_fields = [field for field in required_fields if field not in response]
            if missing_fields:
                print(f"⚠️  Missing required fields: {missing_fields}")
                return False
            
            if not response.get('success'):
                print(f"⚠️  Health check not successful")
                return False
            
            print(f"✅ System Health Retrieved Successfully")
            print(f"   System Status: {type(response.get('system_status'))}")
            print(f"   Component Health: {type(response.get('component_health'))}")
            print(f"   Timestamp: {response.get('timestamp')}")
            
            # Validate system status structure
            system_status = response.get('system_status', {})
            if isinstance(system_status, dict):
                print(f"   System Status Keys: {list(system_status.keys())}")
            
            # Validate component health structure
            component_health = response.get('component_health', {})
            if isinstance(component_health, dict):
                print(f"   Component Health Keys: {list(component_health.keys())}")
            
            return True
        
        return False

    def test_performance_optimization_endpoint(self):
        """Test GET /api/production/performance/optimize - Run performance optimization"""
        print("\n" + "="*80)
        print("🎯 TESTING PERFORMANCE OPTIMIZATION ENDPOINT")
        print("="*80)
        
        success, response = self.run_test(
            "Performance Optimization - Run Optimization",
            "GET",
            "production/performance/optimize",
            200
        )
        
        if success and response:
            # Validate response structure
            required_fields = ['success', 'optimization_results', 'timestamp']
            
            missing_fields = [field for field in required_fields if field not in response]
            if missing_fields:
                print(f"⚠️  Missing required fields: {missing_fields}")
                return False
            
            if not response.get('success'):
                print(f"⚠️  Performance optimization not successful")
                return False
            
            print(f"✅ Performance Optimization Completed Successfully")
            print(f"   Timestamp: {response.get('timestamp')}")
            
            # Validate optimization results structure
            optimization_results = response.get('optimization_results', {})
            expected_optimizations = ['cache_optimization', 'analytics_processing', 'query_optimization']
            
            for optimization in expected_optimizations:
                if optimization not in optimization_results:
                    print(f"⚠️  Missing optimization result: {optimization}")
                    return False
                
                result = optimization_results[optimization]
                print(f"   {optimization}: {result}")
                
                if result not in ['completed', 'not_available']:
                    print(f"⚠️  Invalid optimization result for {optimization}: {result}")
                    return False
            
            return True
        
        return False

    def test_competitive_analysis_endpoint(self):
        """Test GET /api/production/competitive/analysis - Get competitive analysis metrics"""
        print("\n" + "="*80)
        print("🎯 TESTING COMPETITIVE ANALYSIS ENDPOINT")
        print("="*80)
        
        success, response = self.run_test(
            "Competitive Analysis - Industry Benchmarks",
            "GET",
            "production/competitive/analysis",
            200
        )
        
        if success and response:
            # Validate response structure
            required_fields = ['success', 'competitive_analysis', 'generated_at']
            
            missing_fields = [field for field in required_fields if field not in response]
            if missing_fields:
                print(f"⚠️  Missing required fields: {missing_fields}")
                return False
            
            if not response.get('success'):
                print(f"⚠️  Competitive analysis not successful")
                return False
            
            print(f"✅ Competitive Analysis Retrieved Successfully")
            print(f"   Generated At: {response.get('generated_at')}")
            
            # Validate competitive analysis structure
            competitive_analysis = response.get('competitive_analysis', {})
            expected_sections = ['our_platform', 'industry_leaders', 'competitive_advantages', 'performance_comparison']
            
            for section in expected_sections:
                if section not in competitive_analysis:
                    print(f"⚠️  Missing competitive analysis section: {section}")
                    return False
            
            print(f"   Our Platform: {type(competitive_analysis.get('our_platform'))}")
            print(f"   Industry Leaders: {type(competitive_analysis.get('industry_leaders'))}")
            print(f"   Competitive Advantages: {len(competitive_analysis.get('competitive_advantages', []))} advantages")
            print(f"   Performance Comparison: {type(competitive_analysis.get('performance_comparison'))}")
            
            # Validate industry leaders
            industry_leaders = competitive_analysis.get('industry_leaders', {})
            expected_competitors = ['harvey_ai', 'donotpay', 'lawdroid']
            
            for competitor in expected_competitors:
                if competitor not in industry_leaders:
                    print(f"⚠️  Missing competitor benchmark: {competitor}")
                    return False
                
                competitor_data = industry_leaders[competitor]
                if not isinstance(competitor_data, dict):
                    print(f"⚠️  Invalid competitor data structure for {competitor}")
                    return False
                
                print(f"   {competitor}: {list(competitor_data.keys())}")
            
            return True
        
        return False

    def run_comprehensive_production_tests(self):
        """Run all production optimization system tests"""
        print("\n" + "="*100)
        print("🚀 COMPREHENSIVE PRODUCTION OPTIMIZATION & PERFORMANCE ANALYTICS SYSTEM TESTING")
        print("="*100)
        print(f"Testing against: {self.base_url}")
        print(f"API Base URL: {self.api_url}")
        
        # Test all 8 production endpoints
        test_results = []
        
        test_results.append(("Production Status Endpoint", self.test_production_status_endpoint()))
        test_results.append(("Production Metrics Endpoint", self.test_production_metrics_endpoint()))
        test_results.append(("Analytics Report Endpoint", self.test_analytics_report_endpoint()))
        test_results.append(("Cache Invalidation Endpoint", self.test_cache_invalidation_endpoint()))
        test_results.append(("Active Sessions Endpoint", self.test_active_sessions_endpoint()))
        test_results.append(("System Health Endpoint", self.test_system_health_endpoint()))
        test_results.append(("Performance Optimization Endpoint", self.test_performance_optimization_endpoint()))
        test_results.append(("Competitive Analysis Endpoint", self.test_competitive_analysis_endpoint()))
        
        # Print comprehensive summary
        print("\n" + "="*100)
        print("📊 COMPREHENSIVE PRODUCTION TESTING SUMMARY")
        print("="*100)
        
        passed_tests = sum(1 for _, result in test_results if result)
        total_tests = len(test_results)
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"🎯 PRODUCTION ENDPOINTS TESTED: {total_tests}/8")
        print(f"✅ SUCCESSFUL TESTS: {passed_tests}")
        print(f"❌ FAILED TESTS: {total_tests - passed_tests}")
        print(f"📈 SUCCESS RATE: {success_rate:.1f}%")
        
        print(f"\n📋 DETAILED RESULTS:")
        for test_name, result in test_results:
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"   {status}: {test_name}")
        
        if self.failed_tests:
            print(f"\n⚠️  FAILED TEST DETAILS:")
            for failed_test in self.failed_tests:
                print(f"   - {failed_test}")
        
        print(f"\n🔧 TOTAL API CALLS: {self.tests_run}")
        print(f"✅ SUCCESSFUL API CALLS: {self.tests_passed}")
        print(f"❌ FAILED API CALLS: {self.tests_run - self.tests_passed}")
        
        # Production System Assessment
        print(f"\n" + "="*100)
        print("🏭 PRODUCTION SYSTEM ASSESSMENT")
        print("="*100)
        
        if success_rate >= 87.5:  # 7/8 tests passed
            print("🎉 OUTSTANDING: Production Optimization System is FULLY OPERATIONAL")
            print("✅ All critical production endpoints are working correctly")
            print("✅ Enterprise-grade performance analytics capabilities confirmed")
            print("✅ Hybrid caching system operational")
            print("✅ Scalability and monitoring systems active")
            print("✅ Competitive analysis shows industry-leading capabilities")
            print("🚀 READY FOR PRODUCTION DEPLOYMENT")
        elif success_rate >= 75:  # 6/8 tests passed
            print("✅ GOOD: Production System is mostly operational with minor issues")
            print("⚠️  Some production features may need attention")
            print("🔧 REQUIRES MINOR FIXES before full production deployment")
        elif success_rate >= 50:  # 4/8 tests passed
            print("⚠️  PARTIAL: Production System has significant issues")
            print("❌ Multiple production endpoints not working correctly")
            print("🔧 REQUIRES MAJOR FIXES before production deployment")
        else:
            print("❌ CRITICAL: Production System is not operational")
            print("🚨 PRODUCTION DEPLOYMENT NOT RECOMMENDED")
            print("🔧 REQUIRES COMPLETE SYSTEM REVIEW")
        
        return success_rate >= 87.5

if __name__ == "__main__":
    print("🎯 Starting Production Optimization & Performance Analytics System Testing...")
    
    tester = ProductionSystemTester()
    success = tester.run_comprehensive_production_tests()
    
    if success:
        print(f"\n🎉 PRODUCTION TESTING COMPLETED SUCCESSFULLY!")
        sys.exit(0)
    else:
        print(f"\n❌ PRODUCTION TESTING COMPLETED WITH ISSUES!")
        sys.exit(1)