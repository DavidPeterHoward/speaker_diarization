#!/usr/bin/env python3
"""
Meta-Recursive API Endpoints for AudioTranscribe
Provides API endpoints for meta-recursive system management
"""

import asyncio
from typing import Dict, Any
from flask import Blueprint, jsonify, request
from datetime import datetime

from .meta_recursive_integration import get_meta_recursive_integration
from .error_handlers import create_error_response

# Create blueprint
meta_recursive_bp = Blueprint('meta_recursive', __name__)

@meta_recursive_bp.route('/meta-review/trigger', methods=['POST'])
def trigger_meta_review():
    """Trigger a meta-recursive system review"""
    try:
        # Get integration instance
        integration = get_meta_recursive_integration()

        # Trigger review asynchronously
        # Note: In production, this should be handled with background tasks
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            review_result = loop.run_until_complete(integration.perform_system_review())

            if review_result:
                return jsonify({
                    'success': True,
                    'message': 'Meta-recursive review completed',
                    'review_summary': {
                        'overall_health': review_result.meta_analysis.get('overall_health', 0.0),
                        'critical_issues': len(review_result.meta_analysis.get('critical_issues', [])),
                        'recommendations': len(review_result.improvement_plan.get('recommendations', [])),
                        'system_maturity': review_result.meta_analysis.get('system_maturity', 'unknown')
                    }
                }), 200
            else:
                return jsonify({
                    'success': False,
                    'message': 'Meta-recursive review failed'
                }), 500

        finally:
            loop.close()

    except Exception as e:
        import logging
        logging.error(f"Meta-review trigger failed: {e}")
        return create_error_response('Meta-review failed', 500, 'MetaReviewError')

@meta_recursive_bp.route('/meta-review/status', methods=['GET'])
def get_meta_review_status():
    """Get current meta-recursive system status"""
    try:
        integration = get_meta_recursive_integration()

        # Get latest review information
        review_history = integration.meta_reviewer.review_history
        latest_review = review_history[-1] if review_history else None

        status = {
            'system_active': True,
            'continuous_monitoring': integration.continuous_monitoring,
            'review_interval_hours': integration.review_interval_hours,
            'total_reviews': len(review_history),
            'last_review_time': latest_review['timestamp'].isoformat() if latest_review else None,
            'improvements_made': len(integration.improvement_history)
        }

        if latest_review:
            status['latest_review_summary'] = {
                'overall_health': latest_review['results'].meta_analysis.get('overall_health', 0.0),
                'critical_issues': len(latest_review['results'].meta_analysis.get('critical_issues', [])),
                'recommendations': len(latest_review['results'].improvement_plan.get('recommendations', []))
            }

        return jsonify({
            'success': True,
            'status': status
        }), 200

    except Exception as e:
        import logging
        logging.error(f"Meta-review status failed: {e}")
        return create_error_response('Status retrieval failed', 500, 'StatusError')

@meta_recursive_bp.route('/meta-review/history', methods=['GET'])
def get_meta_review_history():
    """Get meta-recursive review history"""
    try:
        integration = get_meta_recursive_integration()
        review_history = integration.meta_reviewer.review_history

        # Convert to serializable format
        history = []
        for review in review_history[-10:]:  # Last 10 reviews
            history.append({
                'timestamp': review['timestamp'].isoformat(),
                'overall_health': review['results'].meta_analysis.get('overall_health', 0.0),
                'critical_issues': len(review['results'].meta_analysis.get('critical_issues', [])),
                'recommendations': len(review['results'].improvement_plan.get('recommendations', [])),
                'system_maturity': review['results'].meta_analysis.get('system_maturity', 'unknown')
            })

        return jsonify({
            'success': True,
            'history': history,
            'total_reviews': len(review_history)
        }), 200

    except Exception as e:
        import logging
        logging.error(f"Meta-review history failed: {e}")
        return create_error_response('History retrieval failed', 500, 'HistoryError')

@meta_recursive_bp.route('/meta-review/improvements', methods=['GET'])
def get_improvement_history():
    """Get improvement implementation history"""
    try:
        integration = get_meta_recursive_integration()
        improvements = integration.improvement_history

        # Convert to serializable format
        improvement_list = []
        for improvement in improvements[-20:]:  # Last 20 improvements
            improvement_list.append({
                'timestamp': improvement['timestamp'],
                'action': improvement['recommendation'].get('action', ''),
                'priority': improvement['recommendation'].get('priority', ''),
                'status': improvement.get('execution_status', 'unknown'),
                'expected_improvement': improvement['recommendation'].get('expected_improvement', '')
            })

        return jsonify({
            'success': True,
            'improvements': improvement_list,
            'total_improvements': len(improvements)
        }), 200

    except Exception as e:
        import logging
        logging.error(f"Improvement history failed: {e}")
        return create_error_response('Improvement history retrieval failed', 500, 'ImprovementHistoryError')

@meta_recursive_bp.route('/meta-review/recommendations', methods=['GET'])
def get_current_recommendations():
    """Get current improvement recommendations"""
    try:
        integration = get_meta_recursive_integration()

        # Get latest review
        review_history = integration.meta_reviewer.review_history
        if not review_history:
            return jsonify({
                'success': True,
                'recommendations': [],
                'message': 'No reviews available yet'
            }), 200

        latest_review = review_history[-1]
        recommendations = latest_review['results'].improvement_plan.get('recommendations', [])

        # Format recommendations
        formatted_recs = []
        for rec in recommendations:
            formatted_recs.append({
                'action': rec.get('action', ''),
                'priority': rec.get('priority', ''),
                'expected_improvement': rec.get('expected_improvement', ''),
                'implementation_effort': rec.get('implementation_effort', ''),
                'timeline': rec.get('timeline', ''),
                'category': rec.get('dimension', 'general')
            })

        return jsonify({
            'success': True,
            'recommendations': formatted_recs,
            'total_recommendations': len(formatted_recs),
            'generated_at': latest_review['timestamp'].isoformat()
        }), 200

    except Exception as e:
        import logging
        logging.error(f"Recommendations retrieval failed: {e}")
        return create_error_response('Recommendations retrieval failed', 500, 'RecommendationsError')

@meta_recursive_bp.route('/meta-review/metrics', methods=['GET'])
def get_meta_metrics():
    """Get comprehensive meta-recursive system metrics"""
    try:
        integration = get_meta_recursive_integration()

        # Gather comprehensive metrics
        metrics = {
            'system_status': {
                'active': True,
                'continuous_monitoring': integration.continuous_monitoring,
                'review_interval_hours': integration.review_interval_hours
            },
            'review_statistics': {
                'total_reviews': len(integration.meta_reviewer.review_history),
                'last_review_time': None,
                'average_health_score': 0.0,
                'improvement_trends': []
            },
            'improvement_statistics': {
                'total_improvements': len(integration.improvement_history),
                'successful_improvements': sum(1 for i in integration.improvement_history if i.get('execution_status') == 'success'),
                'failed_improvements': sum(1 for i in integration.improvement_history if i.get('execution_status') == 'failed'),
                'priority_distribution': {}
            }
        }

        # Calculate review statistics
        if integration.meta_reviewer.review_history:
            reviews = integration.meta_reviewer.review_history
            metrics['review_statistics']['last_review_time'] = reviews[-1]['timestamp'].isoformat()

            health_scores = [
                r['results'].meta_analysis.get('overall_health', 0.0)
                for r in reviews
            ]
            metrics['review_statistics']['average_health_score'] = sum(health_scores) / len(health_scores) if health_scores else 0.0

            # Calculate improvement trends
            if len(reviews) > 1:
                recent_scores = [r['results'].meta_analysis.get('overall_health', 0.0) for r in reviews[-5:]]
                if len(recent_scores) > 1:
                    trend = recent_scores[-1] - recent_scores[0]
                    metrics['review_statistics']['improvement_trends'] = [{
                        'period': 'last_5_reviews',
                        'trend': 'improving' if trend > 0.05 else 'stable' if trend > -0.05 else 'declining',
                        'change': trend
                    }]

        # Calculate improvement priority distribution
        priority_counts = {}
        for improvement in integration.improvement_history:
            priority = improvement['recommendation'].get('priority', 'unknown')
            priority_counts[priority] = priority_counts.get(priority, 0) + 1

        metrics['improvement_statistics']['priority_distribution'] = priority_counts

        return jsonify({
            'success': True,
            'metrics': metrics,
            'timestamp': datetime.utcnow().isoformat()
        }), 200

    except Exception as e:
        import logging
        logging.error(f"Meta metrics retrieval failed: {e}")
        return create_error_response('Metrics retrieval failed', 500, 'MetricsError')

@meta_recursive_bp.route('/meta-review/diagnostics', methods=['GET'])
def get_system_diagnostics():
    """Get detailed system diagnostics for meta-recursive analysis"""
    try:
        integration = get_meta_recursive_integration()

        # Gather system diagnostics
        diagnostics = {
            'reviewer_status': {
                'dimensions_available': list(integration.meta_reviewer.reviewers.keys()),
                'review_history_size': len(integration.meta_reviewer.review_history),
                'last_review_timestamp': None
            },
            'system_health': {
                'continuous_monitoring_active': integration.continuous_monitoring,
                'improvement_execution_enabled': True,  # Could be configurable
                'error_handling_active': True
            },
            'data_quality': {
                'test_data_available': True,  # Check if OpenTTS test data exists
                'benchmark_datasets': ['integrated_test_suite'],
                'validation_active': True
            },
            'performance_indicators': {
                'review_execution_time_avg': 'TBD',  # Would track actual execution times
                'improvement_success_rate': 'TBD',
                'system_adaptation_rate': 'TBD'
            }
        }

        # Update with actual data
        if integration.meta_reviewer.review_history:
            last_review = integration.meta_reviewer.review_history[-1]
            diagnostics['reviewer_status']['last_review_timestamp'] = last_review['timestamp'].isoformat()

        # Check for test data
        from pathlib import Path
        test_data_exists = Path("opentts_transcription_test_report.json").exists()
        diagnostics['data_quality']['test_data_available'] = test_data_exists

        return jsonify({
            'success': True,
            'diagnostics': diagnostics,
            'recommendations': [
                'Ensure continuous monitoring is enabled for optimal performance',
                'Regular review execution helps maintain system health',
                'Monitor improvement success rates for effectiveness validation'
            ]
        }), 200

    except Exception as e:
        import logging
        logging.error(f"System diagnostics failed: {e}")
        return create_error_response('Diagnostics retrieval failed', 500, 'DiagnosticsError')
