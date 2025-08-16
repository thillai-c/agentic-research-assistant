#!/usr/bin/env python3
"""
Agentic Research Assistant UI with Advanced Features
"""

import streamlit as st
import json
import time
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.assistant import AgenticResearchAssistant, ResearchPhase

# ===================================
# Streamlit UI Configuration
# ===================================

st.set_page_config(
    page_title="Agentic Research Assistant",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/agentic-research-assistant',
        'Report a bug': 'https://github.com/your-repo/agentic-research-assistant/issues',
        'About': '# Agentic Research Assistant\nA powerful AI-powered research automation system.'
    }
)

# ===================================
# Custom CSS Styling
# ===================================

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    .phase-indicator {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    
    .agent-status {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
    
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 1rem;
        color: #155724;
    }
    
    .info-box {
        background: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 8px;
        padding: 1rem;
        color: #0c5460;
    }
    
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

# ===================================
# Session State Management
# ===================================

if 'research_assistant' not in st.session_state:
    st.session_state.research_assistant = None
if 'research_results' not in st.session_state:
    st.session_state.research_results = None
if 'research_active' not in st.session_state:
    st.session_state.research_active = False
if 'research_history' not in st.session_state:
    st.session_state.research_history = []
if 'current_topic' not in st.session_state:
    st.session_state.current_topic = ""
if 'show_metrics' not in st.session_state:
    st.session_state.show_metrics = False
if 'show_export' not in st.session_state:
    st.session_state.show_export = False

# ===================================
# Helper Functions
# ===================================

def initialize_assistant():
    """Initialize the agentic research assistant"""
    try:
        if not st.session_state.research_assistant:
            st.session_state.research_assistant = AgenticResearchAssistant()
        return True
    except Exception as e:
        st.error(f"Failed to initialize assistant: {e}")
        return False

def run_research(topic, scope, audience, research_type):
    """Run the agentic research process"""
    try:
        # Initialize assistant if needed
        if not initialize_assistant():
            return
        
        # Update session state
        st.session_state.research_active = True
        st.session_state.current_topic = topic
        
        # Reset Quick Actions states
        st.session_state.show_metrics = False
        st.session_state.show_export = False
        
        # Configure research parameters
        config = {
            "research_depth": research_type,
            "max_sources": 15 if scope == "Comprehensive" else 8,
            "paper_target_length": 8000 if scope == "Comprehensive" else 4000
        }
        
        # Run research
        result = st.session_state.research_assistant.run_research(topic)
        
        # Update session state
        st.session_state.research_results = result
        st.session_state.research_active = False
        
        # Add to history
        if "error" not in result:
            st.session_state.research_history.append({
                "topic": topic,
                "timestamp": datetime.now().isoformat(),
                "completion": result.get('completion_percentage', 0),
                "quality": result.get('metrics', {}).get('quality_score', 0)
            })
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"research_results_{timestamp}.json"
        with open(filename, "w") as f:
            json.dump(result, f, indent=2)
        
        return result
        
    except Exception as e:
        st.error(f"Research failed: {e}")
        st.session_state.research_active = False
        return None

def get_research_status():
    """Get current research status and check if metrics are available"""
    if not st.session_state.research_results:
        return "no_results"
    
    # Check if we have the agentic metrics
    if 'ui_metrics' in st.session_state.research_results:
        return "agentic_metrics"
    elif 'metrics' in st.session_state.research_results:
        return "basic_metrics"
    else:
        return "no_metrics"

def create_progress_chart(phases, percentages):
    """Create a progress chart for research phases"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=phases,
        y=percentages,
        mode='lines+markers',
        name='Progress',
        line=dict(color='#667eea', width=4),
        marker=dict(size=10, color='#667eea')
    ))
    
    fig.update_layout(
        title="Research Progress by Phase",
        xaxis_title="Research Phase",
        yaxis_title="Completion Percentage",
        height=300,
        showlegend=False
    )
    
    return fig

def create_quality_radar_chart(metrics):
    """Create a radar chart for quality metrics"""
    if not metrics:
        return None
    
    categories = ['Sources', 'Data', 'Writing', 'Overall']
    values = [
        metrics.get('avg_relevance', 0) / 10 * 100,
        metrics.get('data_points', 0) / 4 * 100,
        min(100, metrics.get('paper_length', 0) / 1000 * 20),
        metrics.get('quality_score', 0) / 10 * 100
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Quality Metrics',
        line_color='#667eea'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=False,
        height=400
    )
    
    return fig

# ===================================
# Main UI Layout
# ===================================

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚀 Agentic Research Assistant</h1>
        <h3>Advanced AI-Powered Research Automation System</h3>
        <p>Multi-Agent Architecture with Agentic Features</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Research Controls")
        
        # Research Configuration
        st.subheader("Research Settings")
        topic = st.text_area("Research Topic", 
                           placeholder="Enter your research topic here...",
                           height=100,
                           value=st.session_state.current_topic)
        
        scope = st.selectbox("Research Scope", 
                           ["Comprehensive", "Focused", "Quick Review", "Deep Analysis"])
        
        audience = st.selectbox("Target Audience", 
                              ["Academic Researchers", "Students", "Professionals", "General Public"])
        
        research_type = st.selectbox("Research Type",
                                   ["Comprehensive", "Literature Review", "Data Analysis", "Case Study"])
        
        # Advanced Options
        with st.expander("Advanced Options"):
            enable_citations = st.checkbox("Enable Citations", value=True)
            enable_plagiarism_check = st.checkbox("Enable Plagiarism Check", value=True)
            custom_length = st.number_input("Target Paper Length (words)", 
                                          min_value=1000, max_value=15000, value=5000)
        
        # Control Button
        if st.button("🚀 Start Agentic Research", 
                    type="primary", 
                    disabled=st.session_state.research_active,
                    use_container_width=True):
            if topic.strip():
                with st.spinner("Running agentic research..."):
                    result = run_research(topic, scope, audience, research_type)
                    if result:
                        st.success("Agentic research completed successfully!")
                        st.rerun()
            else:
                st.error("Please enter a research topic")
        
        # Status
        st.markdown("---")
        st.subheader("📊 System Status")
        
        if st.session_state.research_active:
            st.success("🔄 Research Active")
        else:
            st.info("⏸️ Research Ready")
        
        # Progress
        if st.session_state.research_results:
            progress = st.session_state.research_results.get('completion_percentage', 0)
            st.progress(progress / 100)
            st.metric("Completion", f"{progress:.1f}%")
            
            # Current Phase
            phase = st.session_state.research_results.get('current_phase', 'unknown')
            phase_icons = {
                'planning': '📋',
                'research': '🔍',
                'analysis': '📊',
                'writing': '✍️',
                'editing': '✏️',
                'complete': '✅'
            }
            st.metric("Current Phase", f"{phase_icons.get(phase, '❓')} {phase.title()}")
        
        # Quick Actions
        st.markdown("---")
        st.subheader("⚡ Quick Actions")
        
        if st.button("📊 View Metrics", use_container_width=True):
            st.session_state.show_metrics = True
        
        if st.button("📚 Export Results", use_container_width=True):
            st.session_state.show_export = True
    
    # Handle Quick Actions
    if st.session_state.get('show_metrics', False):
        st.markdown("---")
        st.header("📊 Comprehensive Research Metrics")
        
        if st.session_state.research_results:
            # Get UI metrics from research assistant
            try:
                # Check research status
                status = get_research_status()
                
                if status == "agentic_metrics":
                    # Use pre-calculated UI metrics
                    ui_metrics = st.session_state.research_results['ui_metrics']
                    metric_summary = st.session_state.research_results.get('metric_summary', {})
                else:
                    # Create a temporary assistant instance to get metrics
                    temp_assistant = AgenticResearchAssistant()
                    temp_assistant.metrics = st.session_state.research_results.get('metrics', {})
                    temp_assistant.literature_sources = st.session_state.research_results.get('literature_sources', [])
                    temp_assistant.data_insights = st.session_state.research_results.get('data_insights', [])
                    temp_assistant.paper_draft = st.session_state.research_results.get('paper_draft', '')
                    temp_assistant.completion_percentage = st.session_state.research_results.get('completion_percentage', 0)
                    
                    # Get comprehensive metrics
                    ui_metrics = temp_assistant.get_ui_metrics()
                    metric_summary = temp_assistant.get_metric_summary()
                
                # Ensure we have the required data
                if not ui_metrics or not metric_summary:
                    st.error("Failed to generate metrics. Please try running the research again.")
                    return
                
                # Display metric summary
                st.subheader("🎯 Key Performance Indicators")
                summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
                
                with summary_col1:
                    st.metric("Overall Quality", f"{ui_metrics['overview']['quality_score']}/10")
                with summary_col2:
                    st.metric("Completion", f"{ui_metrics['overview']['completion_percentage']}%")
                with summary_col3:
                    st.metric("Sources", ui_metrics['overview']['total_sources'])
                with summary_col4:
                    st.metric("Insights", ui_metrics['overview']['data_points'])
                
                # Performance indicators
                st.subheader("🚦 Performance Status")
                perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
                
                # Get performance indicators with fallbacks
                performance_indicators = metric_summary.get('performance_indicators', {})
                
                with perf_col1:
                    source_quality = metric_summary.get('performance_indicators', {}).get('source_quality', '❓')
                    st.markdown(f"**Source Quality:** {source_quality}")
                with perf_col2:
                    temporal_freshness = performance_indicators.get('temporal_freshness', '❓')
                    st.markdown(f"**Temporal Freshness:** {temporal_freshness}")
                with perf_col3:
                    citation_impact = metric_summary.get('performance_indicators', {}).get('citation_impact', '❓')
                    st.markdown(f"**Citation Impact:** {citation_impact}")
                with perf_col4:
                    research_depth = metric_summary.get('performance_indicators', {}).get('research_depth', '❓')
                    st.markdown(f"**Research Depth:** {research_depth}")
                
                # Detailed metrics by category
                st.subheader("📈 Detailed Metrics Breakdown")
                
                # Source Quality Metrics
                with st.expander("🔍 Source Quality Metrics", expanded=True):
                    source_col1, source_col2 = st.columns(2)
                    with source_col1:
                        st.metric("Average Relevance", f"{ui_metrics['source_quality']['avg_relevance']}/10")
                        st.metric("High Relevance Sources", ui_metrics['source_quality']['high_relevance_sources'])
                        st.metric("Source Diversity", f"{ui_metrics['source_quality']['source_diversity_score']}/10")
                    with source_col2:
                        st.metric("Medium Relevance Sources", ui_metrics['source_quality']['medium_relevance_sources'])
                        st.metric("Low Relevance Sources", ui_metrics['source_quality']['low_relevance_sources'])
                        st.metric("Relevance Consistency", f"{ui_metrics['source_quality']['relevance_consistency']:.2f}")
                
                # Citation Metrics
                with st.expander("📚 Citation Metrics"):
                    citation_col1, citation_col2 = st.columns(2)
                    with citation_col1:
                        st.metric("Total Citations", ui_metrics['citations']['total_citations'])
                        st.metric("Average Citations per Source", f"{ui_metrics['citations']['avg_citations_per_source']:.1f}")
                    with citation_col2:
                        st.metric("Highly Cited Sources", ui_metrics['citations']['highly_cited_sources'])
                        st.metric("Citation Impact Score", f"{ui_metrics['citations']['citation_impact_score']}/10")
                
                # Temporal Metrics
                with st.expander("⏰ Temporal Metrics"):
                    temp_col1, temp_col2 = st.columns(2)
                    with temp_col1:
                        st.metric("Recent Sources (3 years)", ui_metrics['temporal']['recent_sources'])
                        st.metric("Current Sources (5 years)", ui_metrics['temporal']['current_sources'])
                    with temp_col2:
                        st.metric("Outdated Sources (10+ years)", ui_metrics['temporal']['outdated_sources'])
                        st.metric("Temporal Freshness", f"{ui_metrics['temporal']['temporal_freshness_score']}/10")
                
                # Data Insight Metrics
                with st.expander("💡 Data Insight Metrics"):
                    insight_col1, insight_col2 = st.columns(2)
                    with insight_col1:
                        st.metric("High Confidence Insights", ui_metrics['insights']['high_confidence_insights'])
                        st.metric("Medium Confidence Insights", ui_metrics['insights']['medium_confidence_insights'])
                    with insight_col2:
                        st.metric("Low Confidence Insights", ui_metrics['insights']['low_confidence_insights'])
                        st.metric("Insight Significance", f"{ui_metrics['insights']['insight_significance_score']}/10")
                        st.metric("Insight Diversity", f"{ui_metrics['insights']['insight_diversity_score']}/10")
                
                # Paper Quality Metrics
                paper_quality = ui_metrics.get('paper_quality', {})
                if paper_quality.get('content_structure_score', 0) > 0:
                    with st.expander("📝 Paper Quality Metrics"):
                        paper_col1, paper_col2 = st.columns(2)
                        with paper_col1:
                            st.metric("Content Structure", f"{paper_quality.get('content_structure_score', 0)}/10")
                            st.metric("Length Adequacy", f"{paper_quality.get('length_adequacy_score', 0)}/10")
                        with paper_col2:
                            st.metric("Writing Quality", f"{paper_quality.get('writing_quality_score', 0)}/10")
                            st.metric("Paper Length", f"{paper_quality.get('paper_length', 0):,} chars")
                
                # Research Depth Metrics
                with st.expander("🔬 Research Depth Metrics"):
                    depth_col1, depth_col2 = st.columns(2)
                    with depth_col1:
                        st.metric("Methodology Coverage", f"{ui_metrics['research_depth']['methodology_coverage_score']}/10")
                        st.metric("Theoretical Framework", f"{ui_metrics['research_depth']['theoretical_framework_score']}/10")
                    with depth_col2:
                        st.metric("Empirical Evidence", f"{ui_metrics['research_depth']['empirical_evidence_score']}/10")
                
                # Overall Assessment
                with st.expander("🎯 Overall Assessment"):
                    assess_col1, assess_col2 = st.columns(2)
                    with assess_col1:
                        st.metric("Comprehensiveness", f"{ui_metrics['assessment']['comprehensiveness_score']}/10")
                        st.metric("Rigor", f"{ui_metrics['assessment']['rigor_score']}/10")
                    with assess_col2:
                        st.metric("Innovation", f"{ui_metrics['assessment']['innovation_score']}/10")
                        st.metric("Impact Potential", f"{ui_metrics['assessment']['impact_potential_score']}/10")
                
                # Recommendations
                st.subheader("💡 Improvement Recommendations")
                recommendations = metric_summary.get('recommendations', [])
                if recommendations:
                    for i, rec in enumerate(recommendations, 1):
                        st.info(f"{i}. {rec}")
                else:
                    st.info("No specific recommendations available at this time.")
                
                # Close metrics view
                if st.button("❌ Close Metrics View", use_container_width=True):
                    st.session_state.show_metrics = False
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Error displaying metrics: {str(e)}")
                st.info("Some metrics may not be available yet. Complete the research process to see all metrics.")
        else:
            st.warning("No research results available. Please run research first to view metrics.")
            if st.button("❌ Close Metrics View", use_container_width=True):
                st.session_state.show_metrics = False
                st.rerun()
    
    if st.session_state.get('show_export', False):
        st.markdown("---")
        st.header("💾 Export Research Results")
        
        if st.session_state.research_results:
            export_col1, export_col2 = st.columns(2)
            
            with export_col1:
                st.subheader("📊 Data Export")
                
                # JSON Export
                json_data = json.dumps(st.session_state.research_results, indent=2)
                st.download_button(
                    label="📥 Download Complete Results (JSON)",
                    data=json_data,
                    file_name=f"agentic_research_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
                
                # Metrics Export
                if 'ui_metrics' in st.session_state.research_results:
                    metrics_data = json.dumps(st.session_state.research_results['ui_metrics'], indent=2)
                    st.download_button(
                        label="📈 Download Metrics Only (JSON)",
                        data=metrics_data,
                        file_name=f"research_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                else:
                    # Fallback to basic metrics
                    basic_metrics = st.session_state.research_results.get('metrics', {})
                    if basic_metrics:
                        metrics_data = json.dumps(basic_metrics, indent=2)
                        st.download_button(
                            label="📈 Download Basic Metrics (JSON)",
                            data=metrics_data,
                            file_name=f"basic_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                            use_container_width=True
                        )
                
                # Summary Export
                summary = {
                    "topic": st.session_state.current_topic,
                    "completion_percentage": st.session_state.research_results.get('completion_percentage'),
                    "current_phase": st.session_state.research_results.get('current_phase'),
                    "sources_count": len(st.session_state.research_results.get('literature_sources', [])),
                    "insights_count": len(st.session_state.research_results.get('data_insights', [])),
                    "paper_length": len(st.session_state.research_results.get('paper_draft', '')),
                    "quality_score": st.session_state.research_results.get('metrics', {}).get('quality_score', 0),
                    "timestamp": datetime.now().isoformat()
                }
                
                st.download_button(
                    label="📋 Download Summary (JSON)",
                    data=json.dumps(summary, indent=2),
                    file_name=f"research_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with export_col2:
                st.subheader("📝 Content Export")
                
                # Paper Export
                if st.session_state.research_results.get('final_paper'):
                    st.download_button(
                        label="📄 Download Final Paper (TXT)",
                        data=st.session_state.research_results['final_paper'],
                        file_name=f"research_paper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                elif st.session_state.research_results.get('paper_draft'):
                    st.download_button(
                        label="📄 Download Paper Draft (TXT)",
                        data=st.session_state.research_results['paper_draft'],
                        file_name=f"paper_draft_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                
                # Sources Export
                if st.session_state.research_results.get('literature_sources'):
                    sources_data = json.dumps(st.session_state.research_results['literature_sources'], indent=2)
                    st.download_button(
                        label="🔍 Download Sources (JSON)",
                        data=sources_data,
                        file_name=f"literature_sources_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                
                # Insights Export
                if st.session_state.research_results.get('data_insights'):
                    insights_data = json.dumps(st.session_state.research_results['data_insights'], indent=2)
                    st.download_button(
                        label="💡 Download Insights (JSON)",
                        data=insights_data,
                        file_name=f"data_insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
            
            # Close export view
            if st.button("❌ Close Export View", use_container_width=True):
                st.session_state.show_export = False
                st.rerun()
        else:
            st.warning("No research results available. Please run research first to export results.")
            if st.button("❌ Close Export View", use_container_width=True):
                st.session_state.show_export = False
                st.rerun()
    
    # Main Content Area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Research Dashboard
        st.header("📊 Agentic Research Dashboard")
        
        if st.session_state.research_results:
            # Progress Overview
            progress_col1, progress_col2, progress_col3, progress_col4 = st.columns(4)
            
            with progress_col1:
                sources = st.session_state.research_results.get('literature_sources', [])
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🔍 Sources</h3>
                    <h2>{len(sources)}</h2>
                    <p>Literature Sources</p>
                </div>
                """, unsafe_allow_html=True)
            
            with progress_col2:
                insights = st.session_state.research_results.get('data_insights', [])
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📊 Insights</h3>
                    <h2>{len(insights)}</h2>
                    <p>Data Points</p>
                </div>
                """, unsafe_allow_html=True)
            
            with progress_col3:
                paper_length = len(st.session_state.research_results.get('paper_draft', ''))
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📝 Length</h3>
                    <h2>{paper_length:,}</h2>
                    <p>Characters</p>
                </div>
                """, unsafe_allow_html=True)
            
            with progress_col4:
                quality = st.session_state.research_results.get('metrics', {}).get('quality_score', 0)
                st.markdown(f"""
                <div class="metric-card">
                    <h3>⭐ Quality</h3>
                    <h2>{quality:.1f}/10</h2>
                    <p>Score</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Progress Chart
            if st.session_state.research_results.get('agent_messages'):
                st.subheader("📈 Research Progress")
                
                phases = ["Planning", "Literature", "Data", "Writing", "Editing"]
                percentages = [10, 30, 50, 75, 100]
                
                fig = create_progress_chart(phases, percentages)
                st.plotly_chart(fig, use_container_width=True)
            
            # Team Status Panel
            st.subheader("👥 Advanced Team Status")
            
            team_col1, team_col2 = st.columns(2)
            
            with team_col1:
                st.markdown("**Research Team**")
                research_status = "🟢 Active" if st.session_state.research_results.get('current_phase') in ['research', 'analysis'] else "⚪ Idle"
                st.info(research_status)
                
                st.markdown("**Content Team**")
                content_status = "🟢 Active" if st.session_state.research_results.get('current_phase') in ['writing', 'editing'] else "⚪ Idle"
                st.info(content_status)
            
            with team_col2:
                st.markdown("**Agent Status**")
                
                # Literature Agent
                lit_status = "✅ Complete" if st.session_state.research_results.get('literature_sources') else "⏳ Pending"
                st.write(f"🔍 Literature: {lit_status}")
                
                # Data Agent
                data_status = "✅ Complete" if st.session_state.research_results.get('data_insights') else "⏳ Pending"
                st.write(f"📊 Data: {data_status}")
                
                # Writer Agent
                writer_status = "✅ Complete" if st.session_state.research_results.get('paper_draft') else "⏳ Pending"
                st.write(f"✍️ Writer: {writer_status}")
                
                # Editor Agent
                editor_status = "✅ Complete" if st.session_state.research_results.get('final_paper') else "⏳ Pending"
                st.write(f"✏️ Editor: {editor_status}")
            
            # Quality Metrics
            metrics = st.session_state.research_results.get('metrics', {})
            if metrics:
                st.subheader("📊 Quality Metrics")
                
                quality_fig = create_quality_radar_chart(metrics)
                if quality_fig:
                    st.plotly_chart(quality_fig, use_container_width=True)
            
            # Agent Activity Monitor
            if st.session_state.research_results.get('agent_messages'):
                st.subheader("📝 Agent Activity Monitor")
                
                for i, message in enumerate(st.session_state.research_results['agent_messages']):
                    with st.expander(f"Agent Message {i+1}", expanded=False):
                        st.write(message)
        else:
            st.info("🚀 Enter a research topic and click 'Start Agentic Research' to begin!")
    
    with col2:
        # Research Outputs
        st.header("📚 Research Outputs")
        
        if st.session_state.research_results:
            # Literature Sources
            sources = st.session_state.research_results.get('literature_sources', [])
            if sources:
                st.subheader("🔍 Literature Sources")
                
                for i, source in enumerate(sources[:3]):  # Show first 3
                    with st.expander(f"Source {i+1}", expanded=False):
                        if isinstance(source, dict):
                            st.write(f"**Title:** {source.get('title', 'N/A')}")
                            st.write(f"**Authors:** {source.get('authors', 'N/A')}")
                            st.write(f"**Year:** {source.get('year', 'N/A')}")
                            st.write(f"**Relevance:** {source.get('relevance_score', 'N/A')}/10")
                            st.write(f"**Summary:** {source.get('summary', 'N/A')[:100]}...")
                        else:
                            st.write(str(source)[:200] + "...")
            
            # Data Insights
            insights = st.session_state.research_results.get('data_insights', [])
            if insights:
                st.subheader("📊 Data Insights")
                
                for i, insight in enumerate(insights[:3]):  # Show first 3
                    with st.expander(f"Insight {i+1}", expanded=False):
                        if isinstance(insight, dict):
                            st.write(f"**Category:** {insight.get('category', 'N/A')}")
                            st.write(f"**Value:** {insight.get('value', 'N/A')}")
                            st.write(f"**Significance:** {insight.get('significance', 'N/A')}")
                            st.write(f"**Confidence:** {insight.get('confidence', 'N/A')}")
                        else:
                            st.write(str(insight)[:200] + "...")
            
            # Paper Preview
            paper_draft = st.session_state.research_results.get('paper_draft', '')
            if paper_draft:
                st.subheader("📝 Paper Preview")
                
                preview_length = 300
                st.text_area("Draft Content", 
                            value=paper_draft[:preview_length] + "..." if len(paper_draft) > preview_length else paper_draft,
                            height=150,
                            disabled=True)
                
                if len(paper_draft) > preview_length:
                    st.info(f"Showing first {preview_length} characters. Full paper available in results.")
            
            # Final Paper
            final_paper = st.session_state.research_results.get('final_paper', '')
            if final_paper:
                st.subheader("✅ Final Paper")
                
                preview_length = 200
                st.text_area("Final Paper", 
                            value=final_paper[:preview_length] + "..." if len(final_paper) > preview_length else final_paper,
                            height=150,
                            disabled=True)
                
                # Download button
                st.download_button(
                    label="📥 Download Final Paper",
                    data=final_paper,
                    file_name=f"agentic_research_paper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
        else:
            st.info("📚 Research outputs will appear here after completion!")
    
    # Bottom Section
    st.markdown("---")
    
    # Research History
    if st.session_state.research_history:
        st.header("📚 Research History")
        
        history_df = pd.DataFrame(st.session_state.research_history)
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
        history_df['date'] = history_df['timestamp'].dt.date
        
        # Create history chart
        fig = px.line(history_df, x='date', y='quality', 
                     title="Research Quality Over Time",
                     labels={'quality': 'Quality Score', 'date': 'Date'})
        st.plotly_chart(fig, use_container_width=True)
        
        # History table
        st.dataframe(history_df[['topic', 'date', 'completion', 'quality']], 
                    use_container_width=True)
    
    # Export Options
    if st.session_state.research_results:
        st.header("💾 Advanced Export Options")
        
        export_col1, export_col2, export_col3, export_col4 = st.columns(4)
        
        with export_col1:
            if st.button("📊 Export as JSON", use_container_width=True):
                st.download_button(
                    label="Download JSON",
                    data=json.dumps(st.session_state.research_results, indent=2),
                    file_name=f"agentic_research_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with export_col2:
            paper = st.session_state.research_results.get('final_paper', '')
            if paper:
                if st.button("📝 Export Paper Only", use_container_width=True):
                    st.download_button(
                        label="Download Paper",
                        data=paper,
                        file_name=f"agentic_research_paper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
        
        with export_col3:
            if st.button("📋 Export Summary", use_container_width=True):
                summary = {
                    "topic": st.session_state.current_topic,
                    "completion_percentage": st.session_state.research_results.get('completion_percentage'),
                    "current_phase": st.session_state.research_results.get('current_phase'),
                    "sources_count": len(st.session_state.research_results.get('literature_sources', [])),
                    "insights_count": len(st.session_state.research_results.get('data_insights', [])),
                    "paper_length": len(st.session_state.research_results.get('paper_draft', '')),
                    "quality_score": st.session_state.research_results.get('metrics', {}).get('quality_score', 0),
                    "timestamp": datetime.now().isoformat()
                }
                
                st.download_button(
                    label="Download Summary",
                    data=json.dumps(summary, indent=2),
                    file_name=f"agentic_research_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with export_col4:
            if st.button("📈 Export Metrics", use_container_width=True):
                metrics = st.session_state.research_results.get('metrics', {})
                if metrics:
                    st.download_button(
                        label="Download Metrics",
                        data=json.dumps(metrics, indent=2),
                        file_name=f"agentic_research_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )

if __name__ == "__main__":
    main()
