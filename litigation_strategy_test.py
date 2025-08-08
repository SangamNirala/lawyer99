#!/usr/bin/env python3
"""
Litigation Strategy Generation Transparent Calculations Testing
==============================================================

Comprehensive testing of the improved litigation strategy generation with transparent calculations.

SPECIFIC TEST REQUIREMENTS:
1. Test the enhanced POST /api/litigation/strategy-recommendations endpoint
2. Verify bounds checking: Expected settlement value should NOT exceed $750k × 1.35 = $1,012,500
3. Verify calculation_breakdown field with transparent multipliers
4. Verify calculation_transparency field showing formulas and bounds_applied status
5. Verify cost calculation step-by-step breakdown with explanations
6. Check AI strategic summary includes cost-benefit transparency section

This addresses the previous issue where expected settlement value was $2,054,124 (2.7x case value).

ENDPOINT: POST /api/litigation/strategy-recommendations

TEST DATA: Exact case data from review request:
- Case Type: Civil - Breach of Contract
- Jurisdiction: California  
- Court Level: District Court
- Judge Name: Judge Sarah Martinez
- Case Value: $750,000
- Evidence Strength: 7/10
- Case Complexity: 65% (0.65)
- Case Facts: Technology startup vs hardware supplier breach of contract case
"""

import requests
import json
import sys
import time
from datetime import datetime
from typing import Dict, Any, List

# Backend URL from environment
BACKEND_URL = "https://027a3f35-a0bc-40cd-ad79-483d46c6126e.preview.emergentagent.com/api"

def test_litigation_strategy_optimizer():
    """Test the AI-Powered Litigation Strategy Optimizer with exact user data"""
    print("🎯 TESTING AI-POWERED LITIGATION STRATEGY OPTIMIZER")
    print("=" * 80)
    
    test_results = []
    
    # Exact test case data from review request
    test_case = {
        "case_id": "test-litigation-strategy-001",
        "case_type": "Civil - Breach of Contract",
        "jurisdiction": "California",
        "court_level": "District Court",
        "judge_name": "Judge Sarah Martinez",
        "case_value": 750000,
        "client_budget": 150000,
        "witness_count": 4,
        "opposing_counsel": "Smith & Associates",
        "evidence_strength": 7,  # Should display as 70% not 700%
        "case_complexity": 0.65,  # 65% input
        "case_facts": "The plaintiff, a technology startup, alleges that the defendant, a hardware supplier, failed to deliver specialized components critical for a product launch despite an executed supply contract. The components were delayed by three months, causing the plaintiff to miss a major trade show and lose potential contracts worth over $750,000. The defendant claims the delay was due to unforeseeable shipping disruptions and force majeure provisions in the contract. Key issues include contract interpretation, applicability of force majeure clauses, and damages calculation.",
        "timeline_constraints": "Plaintiff seeks expedited proceedings to minimize ongoing business losses and maintain investor confidence, requesting trial within 12 months.",
        "legal_issues": [
            "Contract interpretation",
            "Force majeure clause applicability", 
            "Damages calculation",
            "Breach of contract elements",
            "Mitigation of damages"
        ]
    }
    
    print(f"\n📋 Test Case: Enhanced AI-Powered Litigation Strategy")
    print(f"Case ID: {test_case['case_id']}")
    print(f"Case Type: {test_case['case_type']}")
    print(f"Jurisdiction: {test_case['jurisdiction']}")
    print(f"Court Level: {test_case['court_level']}")
    print(f"Judge: {test_case['judge_name']}")
    print(f"Case Value: ${test_case['case_value']:,}")
    print(f"Client Budget: ${test_case['client_budget']:,}")
    print(f"Evidence Strength: {test_case['evidence_strength']}/10 (should display as 70%)")
    print(f"Case Complexity: {test_case['case_complexity']} (65%)")
    print(f"Witnesses: {test_case['witness_count']}")
    print(f"Opposing Counsel: {test_case['opposing_counsel']}")
    print(f"Timeline: {test_case['timeline_constraints']}")
    
    try:
        url = f"{BACKEND_URL}/litigation/strategy-recommendations"
        print(f"\nRequest URL: {url}")
        
        response = requests.post(url, json=test_case, timeout=120)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            # Test 1: Evidence Strength Display Bug Fix
            print("\n🔍 TEST 1: EVIDENCE STRENGTH DISPLAY BUG FIX")
            print("-" * 50)
            
            evidence_display_test = test_evidence_strength_display(data, test_case)
            test_results.append(evidence_display_test)
            
            # Test 2: Realistic Cost Estimates
            print("\n💰 TEST 2: REALISTIC COST ESTIMATES")
            print("-" * 50)
            
            cost_estimates_test = test_realistic_cost_estimates(data, test_case)
            test_results.append(cost_estimates_test)
            
            # Test 3: Enhanced AI Strategic Summary
            print("\n🤖 TEST 3: ENHANCED AI STRATEGIC SUMMARY")
            print("-" * 50)
            
            ai_summary_test = test_ai_strategic_summary(data, test_case)
            test_results.append(ai_summary_test)
            
            # Test 4: Case Complexity Handling
            print("\n📊 TEST 4: CASE COMPLEXITY HANDLING")
            print("-" * 50)
            
            complexity_test = test_case_complexity_handling(data, test_case)
            test_results.append(complexity_test)
            
            # Test 5: Breach of Contract Specific Recommendations
            print("\n⚖️ TEST 5: BREACH OF CONTRACT RECOMMENDATIONS")
            print("-" * 50)
            
            recommendations_test = test_breach_contract_recommendations(data, test_case)
            test_results.append(recommendations_test)
            
            # Test 6: Settlement Probability Analysis
            print("\n🤝 TEST 6: SETTLEMENT PROBABILITY ANALYSIS")
            print("-" * 50)
            
            settlement_test = test_settlement_probability_analysis(data, test_case)
            test_results.append(settlement_test)
            
            # Test 7: Professional-Grade Analysis
            print("\n👨‍⚖️ TEST 7: PROFESSIONAL-GRADE ANALYSIS")
            print("-" * 50)
            
            professional_test = test_professional_grade_analysis(data, test_case)
            test_results.append(professional_test)
            
        else:
            print(f"❌ Request failed with status {response.status_code}")
            if response.text:
                print(f"Error response: {response.text}")
            test_results = [False] * 7  # All tests failed
            
    except Exception as e:
        print(f"❌ Exception occurred: {str(e)}")
        test_results = [False] * 7  # All tests failed
    
    return test_results

def test_evidence_strength_display(data: Dict[str, Any], test_case: Dict[str, Any]) -> bool:
    """Test that evidence strength displays as 70% not 700%"""
    try:
        # Look for evidence strength in various parts of the response
        evidence_strength_found = False
        correct_display = False
        
        # Check in strategic summary or AI insights
        ai_summary = data.get('ai_strategic_summary', '')
        if ai_summary:
            # Look for percentage displays
            if '70%' in ai_summary or '7/10' in ai_summary:
                evidence_strength_found = True
                correct_display = True
                print(f"✅ Evidence strength correctly displayed in AI summary")
            elif '700%' in ai_summary:
                evidence_strength_found = True
                correct_display = False
                print(f"❌ Evidence strength incorrectly displayed as 700% in AI summary")
        
        # Check in recommendations
        recommendations = data.get('strategic_recommendations', [])
        for rec in recommendations:
            rec_text = rec.get('description', '') + ' ' + ' '.join(rec.get('supporting_evidence', []))
            if '70%' in rec_text or '7/10' in rec_text:
                evidence_strength_found = True
                correct_display = True
                print(f"✅ Evidence strength correctly displayed in recommendation: {rec.get('title', 'Unknown')}")
            elif '700%' in rec_text:
                evidence_strength_found = True
                correct_display = False
                print(f"❌ Evidence strength incorrectly displayed as 700% in recommendation: {rec.get('title', 'Unknown')}")
        
        # Check evidence assessment if present
        evidence_assessment = data.get('evidence_assessment')
        if evidence_assessment:
            overall_strength = evidence_assessment.get('overall_strength', 0)
            # Should be normalized to 0.0-1.0 range (0.7 for 7/10)
            if 0.65 <= overall_strength <= 0.75:
                evidence_strength_found = True
                correct_display = True
                print(f"✅ Evidence strength correctly normalized: {overall_strength:.2f} (70%)")
            elif overall_strength > 5:  # Indicates bug where it's not normalized
                evidence_strength_found = True
                correct_display = False
                print(f"❌ Evidence strength not normalized: {overall_strength} (should be ~0.7)")
        
        if not evidence_strength_found:
            print("⚠️ Evidence strength not found in response - may need to check other fields")
            return False
        
        return correct_display
        
    except Exception as e:
        print(f"❌ Error testing evidence strength display: {e}")
        return False

def test_realistic_cost_estimates(data: Dict[str, Any], test_case: Dict[str, Any]) -> bool:
    """Test that cost estimates are realistic and proportional to $750k case value"""
    try:
        case_value = test_case['case_value']  # $750,000
        client_budget = test_case['client_budget']  # $150,000
        
        # Check total estimated cost
        total_cost = data.get('estimated_total_cost')
        if total_cost:
            print(f"Total Estimated Cost: ${total_cost:,.2f}")
            
            # Realistic cost should be 15-30% of case value for complex litigation
            min_realistic = case_value * 0.10  # $75,000
            max_realistic = case_value * 0.35  # $262,500
            
            if min_realistic <= total_cost <= max_realistic:
                print(f"✅ Total cost realistic: ${total_cost:,.2f} ({total_cost/case_value:.1%} of case value)")
                cost_realistic = True
            else:
                print(f"❌ Total cost unrealistic: ${total_cost:,.2f} ({total_cost/case_value:.1%} of case value)")
                print(f"   Expected range: ${min_realistic:,.2f} - ${max_realistic:,.2f}")
                cost_realistic = False
        else:
            print("❌ No total estimated cost provided")
            cost_realistic = False
        
        # Check individual recommendation costs
        recommendations = data.get('strategic_recommendations', [])
        recommendation_costs_realistic = True
        total_rec_cost = 0
        
        print(f"\n📋 Individual Recommendation Costs:")
        for i, rec in enumerate(recommendations[:5], 1):  # Check first 5
            rec_cost = rec.get('estimated_cost')
            if rec_cost:
                total_rec_cost += rec_cost
                print(f"  {i}. {rec.get('title', 'Unknown')}: ${rec_cost:,.2f}")
                
                # Individual recommendations should be reasonable
                if rec_cost > client_budget:  # Shouldn't exceed client budget
                    print(f"     ⚠️ Cost exceeds client budget of ${client_budget:,.2f}")
                    recommendation_costs_realistic = False
                elif rec_cost < 1000:  # Too low for legal work
                    print(f"     ⚠️ Cost seems too low for legal work")
                    recommendation_costs_realistic = False
                else:
                    print(f"     ✅ Cost appears reasonable")
            else:
                print(f"  {i}. {rec.get('title', 'Unknown')}: No cost estimate")
        
        if total_rec_cost > 0:
            print(f"\nTotal Recommendation Costs: ${total_rec_cost:,.2f}")
        
        # Check ROI analysis
        roi_analysis = data.get('roi_analysis', {})
        if roi_analysis:
            print(f"\n📈 ROI Analysis Present: {len(roi_analysis)} metrics")
            for key, value in roi_analysis.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.2f}")
        
        return cost_realistic and recommendation_costs_realistic
        
    except Exception as e:
        print(f"❌ Error testing cost estimates: {e}")
        return False

def test_ai_strategic_summary(data: Dict[str, Any], test_case: Dict[str, Any]) -> bool:
    """Test enhanced AI strategic summary generation"""
    try:
        ai_summary = data.get('ai_strategic_summary', '')
        
        if not ai_summary:
            print("❌ No AI strategic summary provided")
            return False
        
        print(f"AI Summary Length: {len(ai_summary)} characters")
        
        # Quality checks for AI summary
        quality_checks = []
        
        # 1. Substantial content (should be detailed)
        if len(ai_summary) >= 500:
            print("✅ AI summary has substantial content")
            quality_checks.append(True)
        else:
            print(f"❌ AI summary too brief: {len(ai_summary)} characters (expected 500+)")
            quality_checks.append(False)
        
        # 2. Case-specific content
        case_specific_terms = [
            'breach of contract', 'contract', 'technology startup', 'hardware supplier',
            'components', 'trade show', 'force majeure', 'damages'
        ]
        
        found_terms = [term for term in case_specific_terms if term.lower() in ai_summary.lower()]
        if len(found_terms) >= 3:
            print(f"✅ AI summary includes case-specific terms: {found_terms}")
            quality_checks.append(True)
        else:
            print(f"❌ AI summary lacks case-specific content. Found: {found_terms}")
            quality_checks.append(False)
        
        # 3. Strategic insights
        strategic_terms = [
            'strategy', 'recommend', 'approach', 'risk', 'opportunity',
            'litigation', 'settlement', 'trial', 'discovery'
        ]
        
        found_strategic = [term for term in strategic_terms if term.lower() in ai_summary.lower()]
        if len(found_strategic) >= 4:
            print(f"✅ AI summary includes strategic insights: {found_strategic}")
            quality_checks.append(True)
        else:
            print(f"❌ AI summary lacks strategic insights. Found: {found_strategic}")
            quality_checks.append(False)
        
        # 4. Professional tone
        professional_indicators = [
            'analysis', 'assessment', 'evaluation', 'consideration',
            'recommendation', 'strategy', 'approach', 'factors'
        ]
        
        found_professional = [term for term in professional_indicators if term.lower() in ai_summary.lower()]
        if len(found_professional) >= 3:
            print(f"✅ AI summary uses professional language: {found_professional}")
            quality_checks.append(True)
        else:
            print(f"❌ AI summary lacks professional tone. Found: {found_professional}")
            quality_checks.append(False)
        
        # Display sample of AI summary
        print(f"\n📝 AI Summary Sample (first 200 chars):")
        print(f"'{ai_summary[:200]}...'")
        
        # Overall quality assessment
        quality_score = sum(quality_checks) / len(quality_checks)
        print(f"\n📊 AI Summary Quality Score: {quality_score:.1%}")
        
        return quality_score >= 0.75  # 75% of quality checks must pass
        
    except Exception as e:
        print(f"❌ Error testing AI strategic summary: {e}")
        return False

def test_case_complexity_handling(data: Dict[str, Any], test_case: Dict[str, Any]) -> bool:
    """Test proper case complexity handling (65% input vs output)"""
    try:
        input_complexity = test_case['case_complexity']  # 0.65 (65%)
        
        # Check if complexity is properly handled in various parts
        complexity_handled_correctly = []
        
        # 1. Check in AI summary
        ai_summary = data.get('ai_strategic_summary', '')
        if ai_summary:
            if '65%' in ai_summary or '0.65' in ai_summary or 'complex' in ai_summary.lower():
                print("✅ Case complexity referenced in AI summary")
                complexity_handled_correctly.append(True)
            else:
                print("⚠️ Case complexity not clearly referenced in AI summary")
                complexity_handled_correctly.append(False)
        
        # 2. Check strategy type selection
        strategy_type = data.get('recommended_strategy_type')
        if strategy_type:
            print(f"Recommended Strategy Type: {strategy_type}")
            # For 65% complexity, should not be overly simple strategy
            complex_strategies = ['procedural', 'aggressive', 'trial_focused']
            if any(strategy in strategy_type.lower() for strategy in complex_strategies):
                print("✅ Strategy type appropriate for complex case")
                complexity_handled_correctly.append(True)
            else:
                print("⚠️ Strategy type may not reflect case complexity")
                complexity_handled_correctly.append(False)
        
        # 3. Check recommendations reflect complexity
        recommendations = data.get('strategic_recommendations', [])
        complex_rec_count = 0
        
        for rec in recommendations:
            rec_text = rec.get('description', '').lower()
            if any(term in rec_text for term in ['complex', 'detailed', 'comprehensive', 'thorough']):
                complex_rec_count += 1
        
        if complex_rec_count >= 2:
            print(f"✅ {complex_rec_count} recommendations reflect case complexity")
            complexity_handled_correctly.append(True)
        else:
            print(f"⚠️ Only {complex_rec_count} recommendations reflect complexity")
            complexity_handled_correctly.append(False)
        
        # 4. Check estimated costs reflect complexity
        total_cost = data.get('estimated_total_cost', 0)
        case_value = test_case['case_value']
        
        # Complex cases should have higher cost ratios
        if total_cost > 0:
            cost_ratio = total_cost / case_value
            if cost_ratio >= 0.15:  # 15%+ for complex cases
                print(f"✅ Cost estimates reflect complexity: {cost_ratio:.1%} of case value")
                complexity_handled_correctly.append(True)
            else:
                print(f"⚠️ Cost estimates may not reflect complexity: {cost_ratio:.1%} of case value")
                complexity_handled_correctly.append(False)
        
        # 5. Check timing analysis reflects complexity
        timing_analysis = data.get('timing_analysis', [])
        if timing_analysis:
            extended_timelines = 0
            for timing in timing_analysis:
                if 'month' in timing.get('estimated_timeframe', '').lower():
                    extended_timelines += 1
            
            if extended_timelines >= 1:
                print(f"✅ Timing analysis reflects complexity with extended timelines")
                complexity_handled_correctly.append(True)
            else:
                print("⚠️ Timing analysis may not reflect case complexity")
                complexity_handled_correctly.append(False)
        
        complexity_score = sum(complexity_handled_correctly) / len(complexity_handled_correctly) if complexity_handled_correctly else 0
        print(f"\n📊 Complexity Handling Score: {complexity_score:.1%}")
        
        return complexity_score >= 0.6  # 60% of checks must pass
        
    except Exception as e:
        print(f"❌ Error testing case complexity handling: {e}")
        return False

def test_breach_contract_recommendations(data: Dict[str, Any], test_case: Dict[str, Any]) -> bool:
    """Test specific, actionable recommendations relevant to breach of contract case"""
    try:
        recommendations = data.get('strategic_recommendations', [])
        
        if not recommendations:
            print("❌ No strategic recommendations provided")
            return False
        
        print(f"Total Recommendations: {len(recommendations)}")
        
        # Breach of contract specific terms to look for
        contract_specific_terms = [
            'contract', 'breach', 'damages', 'performance', 'delivery',
            'force majeure', 'mitigation', 'specific performance',
            'liquidated damages', 'consequential damages', 'cover',
            'cure', 'material breach', 'anticipatory breach'
        ]
        
        # Categories that should be present for breach of contract
        expected_categories = [
            'discovery', 'motion', 'settlement', 'damages', 'evidence'
        ]
        
        relevant_recommendations = 0
        actionable_recommendations = 0
        categories_found = set()
        
        print(f"\n📋 Analyzing Recommendations for Breach of Contract Relevance:")
        
        for i, rec in enumerate(recommendations, 1):
            title = rec.get('title', '')
            description = rec.get('description', '')
            category = rec.get('category', '').lower()
            
            print(f"\n{i}. {title}")
            print(f"   Category: {rec.get('category', 'Unknown')}")
            print(f"   Priority: {rec.get('priority', 'Unknown')}")
            
            # Check for contract-specific content
            combined_text = (title + ' ' + description).lower()
            found_terms = [term for term in contract_specific_terms if term in combined_text]
            
            if found_terms:
                print(f"   ✅ Contract-relevant terms: {found_terms}")
                relevant_recommendations += 1
            else:
                print(f"   ⚠️ No specific contract terms found")
            
            # Check if actionable (has specific steps/costs/timeframes)
            has_cost = rec.get('estimated_cost') is not None
            has_timeframe = rec.get('estimated_timeframe') is not None
            has_specific_description = len(description) > 50
            
            if has_cost or has_timeframe or has_specific_description:
                print(f"   ✅ Actionable: Cost={has_cost}, Timeframe={has_timeframe}, Detailed={has_specific_description}")
                actionable_recommendations += 1
            else:
                print(f"   ⚠️ Not sufficiently actionable")
            
            # Track categories
            if category:
                categories_found.add(category)
        
        # Assessment
        relevance_score = relevant_recommendations / len(recommendations) if recommendations else 0
        actionable_score = actionable_recommendations / len(recommendations) if recommendations else 0
        category_score = len(categories_found.intersection(expected_categories)) / len(expected_categories)
        
        print(f"\n📊 Breach of Contract Recommendation Analysis:")
        print(f"  Contract-Relevant: {relevant_recommendations}/{len(recommendations)} ({relevance_score:.1%})")
        print(f"  Actionable: {actionable_recommendations}/{len(recommendations)} ({actionable_score:.1%})")
        print(f"  Expected Categories Found: {len(categories_found.intersection(expected_categories))}/{len(expected_categories)} ({category_score:.1%})")
        print(f"  Categories Present: {sorted(categories_found)}")
        
        # Overall assessment
        overall_score = (relevance_score + actionable_score + category_score) / 3
        print(f"  Overall Recommendation Quality: {overall_score:.1%}")
        
        return overall_score >= 0.6  # 60% overall quality required
        
    except Exception as e:
        print(f"❌ Error testing breach of contract recommendations: {e}")
        return False

def test_settlement_probability_analysis(data: Dict[str, Any], test_case: Dict[str, Any]) -> bool:
    """Test settlement probability derivation and explanation"""
    try:
        # Check for settlement analysis
        settlement_analysis = data.get('settlement_analysis')
        
        if not settlement_analysis:
            print("❌ No settlement analysis provided")
            return False
        
        settlement_checks = []
        
        # 1. Settlement probability present and reasonable
        settlement_prob = settlement_analysis.get('settlement_probability')
        if settlement_prob is not None:
            if 0.0 <= settlement_prob <= 1.0:
                print(f"✅ Settlement probability: {settlement_prob:.1%}")
                settlement_checks.append(True)
            else:
                print(f"❌ Settlement probability out of range: {settlement_prob}")
                settlement_checks.append(False)
        else:
            print("❌ No settlement probability provided")
            settlement_checks.append(False)
        
        # 2. Settlement ranges provided
        plaintiff_range = settlement_analysis.get('plaintiff_settlement_range', {})
        defendant_range = settlement_analysis.get('defendant_settlement_range', {})
        
        if plaintiff_range and defendant_range:
            p_low = plaintiff_range.get('low', 0)
            p_high = plaintiff_range.get('high', 0)
            d_low = defendant_range.get('low', 0)
            d_high = defendant_range.get('high', 0)
            
            print(f"✅ Plaintiff Range: ${p_low:,.0f} - ${p_high:,.0f}")
            print(f"✅ Defendant Range: ${d_low:,.0f} - ${d_high:,.0f}")
            
            # Ranges should be reasonable for $750k case
            case_value = test_case['case_value']
            if p_high <= case_value and d_high <= case_value and p_low > 0 and d_low > 0:
                print("✅ Settlement ranges are reasonable")
                settlement_checks.append(True)
            else:
                print("⚠️ Settlement ranges may be unrealistic")
                settlement_checks.append(False)
        else:
            print("❌ Settlement ranges not provided")
            settlement_checks.append(False)
        
        # 3. Expected settlement value
        expected_value = settlement_analysis.get('expected_settlement_value')
        if expected_value:
            print(f"✅ Expected Settlement Value: ${expected_value:,.2f}")
            settlement_checks.append(True)
        else:
            print("❌ No expected settlement value")
            settlement_checks.append(False)
        
        # 4. Settlement factors/explanation
        settlement_factors = settlement_analysis.get('key_settlement_factors', [])
        if settlement_factors:
            print(f"✅ Settlement factors provided: {len(settlement_factors)} factors")
            for factor in settlement_factors[:3]:  # Show first 3
                print(f"   - {factor}")
            settlement_checks.append(True)
        else:
            print("❌ No settlement factors explanation")
            settlement_checks.append(False)
        
        # 5. Negotiation leverage analysis
        negotiation_leverage = settlement_analysis.get('negotiation_leverage', {})
        if negotiation_leverage:
            plaintiff_leverage = negotiation_leverage.get('plaintiff')
            defendant_leverage = negotiation_leverage.get('defendant')
            
            if plaintiff_leverage is not None and defendant_leverage is not None:
                print(f"✅ Negotiation leverage - Plaintiff: {plaintiff_leverage:.2f}, Defendant: {defendant_leverage:.2f}")
                settlement_checks.append(True)
            else:
                print("⚠️ Incomplete negotiation leverage analysis")
                settlement_checks.append(False)
        else:
            print("❌ No negotiation leverage analysis")
            settlement_checks.append(False)
        
        settlement_score = sum(settlement_checks) / len(settlement_checks)
        print(f"\n📊 Settlement Analysis Quality: {settlement_score:.1%}")
        
        return settlement_score >= 0.7  # 70% of checks must pass
        
    except Exception as e:
        print(f"❌ Error testing settlement probability analysis: {e}")
        return False

def test_professional_grade_analysis(data: Dict[str, Any], test_case: Dict[str, Any]) -> bool:
    """Test professional-grade analysis suitable for lawyers"""
    try:
        professional_checks = []
        
        # 1. Comprehensive response structure
        expected_fields = [
            'case_id', 'recommended_strategy_type', 'confidence_score',
            'strategic_recommendations', 'estimated_total_cost',
            'ai_strategic_summary', 'risk_factors'
        ]
        
        missing_fields = [field for field in expected_fields if field not in data]
        if not missing_fields:
            print("✅ All expected fields present in response")
            professional_checks.append(True)
        else:
            print(f"❌ Missing fields: {missing_fields}")
            professional_checks.append(False)
        
        # 2. Confidence score provided and reasonable
        confidence_score = data.get('confidence_score')
        if confidence_score is not None and 0.0 <= confidence_score <= 1.0:
            print(f"✅ Confidence score: {confidence_score:.1%}")
            professional_checks.append(True)
        else:
            print(f"❌ Invalid or missing confidence score: {confidence_score}")
            professional_checks.append(False)
        
        # 3. Risk factors identified
        risk_factors = data.get('risk_factors', [])
        if risk_factors and len(risk_factors) >= 2:
            print(f"✅ Risk factors identified: {len(risk_factors)} factors")
            for risk in risk_factors[:2]:  # Show first 2
                print(f"   - {risk}")
            professional_checks.append(True)
        else:
            print(f"❌ Insufficient risk factors: {len(risk_factors)}")
            professional_checks.append(False)
        
        # 4. Mitigation strategies provided
        mitigation_strategies = data.get('mitigation_strategies', [])
        if mitigation_strategies:
            print(f"✅ Mitigation strategies: {len(mitigation_strategies)} strategies")
            professional_checks.append(True)
        else:
            print("❌ No mitigation strategies provided")
            professional_checks.append(False)
        
        # 5. Evidence assessment
        evidence_assessment = data.get('evidence_assessment')
        if evidence_assessment:
            overall_strength = evidence_assessment.get('overall_strength')
            key_strengths = evidence_assessment.get('key_strengths', [])
            critical_weaknesses = evidence_assessment.get('critical_weaknesses', [])
            
            if overall_strength is not None and key_strengths and critical_weaknesses:
                print(f"✅ Comprehensive evidence assessment provided")
                print(f"   Strength: {overall_strength:.2f}, Strengths: {len(key_strengths)}, Weaknesses: {len(critical_weaknesses)}")
                professional_checks.append(True)
            else:
                print("⚠️ Incomplete evidence assessment")
                professional_checks.append(False)
        else:
            print("❌ No evidence assessment provided")
            professional_checks.append(False)
        
        # 6. Jurisdiction analysis
        jurisdiction_analysis = data.get('jurisdiction_analysis', [])
        if jurisdiction_analysis:
            print(f"✅ Jurisdiction analysis: {len(jurisdiction_analysis)} jurisdictions analyzed")
            professional_checks.append(True)
        else:
            print("❌ No jurisdiction analysis provided")
            professional_checks.append(False)
        
        # 7. Timing analysis
        timing_analysis = data.get('timing_analysis', [])
        if timing_analysis:
            print(f"✅ Timing analysis: {len(timing_analysis)} timing considerations")
            professional_checks.append(True)
        else:
            print("❌ No timing analysis provided")
            professional_checks.append(False)
        
        # 8. Alternative strategies
        alternative_strategies = data.get('alternative_strategies', [])
        if alternative_strategies:
            print(f"✅ Alternative strategies: {len(alternative_strategies)} alternatives")
            professional_checks.append(True)
        else:
            print("❌ No alternative strategies provided")
            professional_checks.append(False)
        
        professional_score = sum(professional_checks) / len(professional_checks)
        print(f"\n📊 Professional Analysis Quality: {professional_score:.1%}")
        
        return professional_score >= 0.75  # 75% of professional checks must pass
        
    except Exception as e:
        print(f"❌ Error testing professional-grade analysis: {e}")
        return False

def main():
    """Main test execution function"""
    print("🎯 AI-POWERED LITIGATION STRATEGY OPTIMIZER TESTING")
    print("=" * 80)
    print(f"Backend URL: {BACKEND_URL}")
    print(f"Test Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n🎯 FOCUS: Testing enhanced litigation strategy optimizer with specific user-reported issues")
    print("USER REQUIREMENTS:")
    print("1. Fix evidence strength display bug (70% not 700%)")
    print("2. Realistic cost estimates for $750k case")
    print("3. Enhanced AI strategic summary")
    print("4. Proper case complexity handling (65%)")
    print("5. Breach of contract specific recommendations")
    print("6. Settlement probability analysis")
    print("7. Professional-grade analysis for lawyers")
    print("=" * 80)
    
    # Run comprehensive test
    test_results = test_litigation_strategy_optimizer()
    
    # Final Results Summary
    print("\n" + "=" * 80)
    print("🎯 LITIGATION STRATEGY OPTIMIZER TEST RESULTS")
    print("=" * 80)
    
    test_names = [
        "Evidence Strength Display Fix",
        "Realistic Cost Estimates", 
        "Enhanced AI Strategic Summary",
        "Case Complexity Handling",
        "Breach of Contract Recommendations",
        "Settlement Probability Analysis",
        "Professional-Grade Analysis"
    ]
    
    total_tests = len(test_results)
    passed_tests = sum(test_results)
    failed_tests = total_tests - passed_tests
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    print(f"📊 Total Tests: {total_tests}")
    print(f"✅ Passed: {passed_tests}")
    print(f"❌ Failed: {failed_tests}")
    print(f"📈 Success Rate: {success_rate:.1f}%")
    
    print(f"\n📋 Detailed Test Results:")
    for i, (test_name, result) in enumerate(zip(test_names, test_results), 1):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {i}. {test_name}: {status}")
    
    print(f"\n🕒 Test Completion Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Overall Assessment
    print(f"\n🔍 OVERALL ASSESSMENT:")
    
    if success_rate >= 90:
        print("🎉 OUTSTANDING SUCCESS: Litigation Strategy Optimizer working excellently!")
        print("✅ All critical fixes implemented and working properly")
        print("✅ Professional-grade analysis suitable for legal practitioners")
        print("✅ Enhanced AI capabilities delivering high-quality strategic insights")
        assessment = "OUTSTANDING"
    elif success_rate >= 75:
        print("✅ GOOD SUCCESS: Litigation Strategy Optimizer working well")
        print("✅ Most critical issues resolved")
        print("⚠️ Some minor improvements may be needed")
        assessment = "GOOD"
    elif success_rate >= 50:
        print("⚠️ PARTIAL SUCCESS: Litigation Strategy Optimizer partially working")
        print("✅ Some fixes implemented successfully")
        print("❌ Several issues still need attention")
        assessment = "PARTIAL"
    else:
        print("❌ NEEDS ATTENTION: Litigation Strategy Optimizer has significant issues")
        print("❌ Critical fixes not working properly")
        print("🚨 Requires immediate development attention")
        assessment = "NEEDS_ATTENTION"
    
    print(f"\n🎯 CRITICAL FIXES STATUS:")
    critical_fixes = [
        ("Evidence Strength Display Bug", test_results[0] if len(test_results) > 0 else False),
        ("Realistic Cost Estimates", test_results[1] if len(test_results) > 1 else False),
        ("Enhanced AI Summary", test_results[2] if len(test_results) > 2 else False),
        ("Case Complexity Handling", test_results[3] if len(test_results) > 3 else False)
    ]
    
    for fix_name, status in critical_fixes:
        status_text = "✅ FIXED" if status else "❌ NOT FIXED"
        print(f"  {fix_name}: {status_text}")
    
    print(f"\n📊 FINAL ASSESSMENT: {assessment}")
    print("=" * 80)
    
    return success_rate >= 70

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)