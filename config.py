#!/usr/bin/env python3
"""
Configuration file for Agentic Research Assistant
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the Agentic Research Assistant"""
    
    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    
    # Research Configuration
    RESEARCH_DEPTH = os.getenv("RESEARCH_DEPTH", "comprehensive")
    MAX_SOURCES = int(os.getenv("MAX_SOURCES", "15"))
    PAPER_TARGET_LENGTH = int(os.getenv("PAPER_TARGET_LENGTH", "5000"))
    
    # Agent Configuration
    ENABLE_CITATIONS = os.getenv("ENABLE_CITATIONS", "true").lower() == "true"
    ENABLE_PLAGIARISM_CHECK = os.getenv("ENABLE_PLAGIARISM_CHECK", "true").lower() == "true"
    
    # UI Configuration
    STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))
    STREAMLIT_HOST = os.getenv("STREAMLIT_HOST", "localhost")
    
    # LLM Configuration
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4000"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
    
    # Search Configuration
    TAVILY_MAX_RESULTS = int(os.getenv("TAVILY_MAX_RESULTS", "10"))
    SEARCH_TIMEOUT = int(os.getenv("SEARCH_TIMEOUT", "30"))
    
    # Performance Configuration
    ENABLE_CACHING = os.getenv("ENABLE_CACHING", "true").lower() == "true"
    CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour
    MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "5"))
    
    # Export Configuration
    EXPORT_FORMATS = ["json", "txt", "pdf", "docx"]
    DEFAULT_EXPORT_FORMAT = os.getenv("DEFAULT_EXPORT_FORMAT", "json")
    
    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "agentic_research.log")
    ENABLE_DEBUG = os.getenv("ENABLE_DEBUG", "false").lower() == "true"
    
    @classmethod
    def validate(cls) -> Dict[str, Any]:
        """Validate configuration and return any issues"""
        issues = {}
        
        # Check required API keys
        if not cls.GROQ_API_KEY:
            issues["GROQ_API_KEY"] = "Missing GROQ API key"
        
        if not cls.TAVILY_API_KEY:
            issues["TAVILY_API_KEY"] = "Missing Tavily API key"
        
        # Validate numeric values
        if cls.MAX_SOURCES < 1 or cls.MAX_SOURCES > 50:
            issues["MAX_SOURCES"] = "MAX_SOURCES must be between 1 and 50"
        
        if cls.PAPER_TARGET_LENGTH < 1000 or cls.PAPER_TARGET_LENGTH > 20000:
            issues["PAPER_TARGET_LENGTH"] = "PAPER_TARGET_LENGTH must be between 1000 and 20000"
        
        if cls.STREAMLIT_PORT < 1024 or cls.STREAMLIT_PORT > 65535:
            issues["STREAMLIT_PORT"] = "STREAMLIT_PORT must be between 1024 and 65535"
        
        return issues
    
    @classmethod
    def get_research_config(cls) -> Dict[str, Any]:
        """Get research-specific configuration"""
        return {
            "research_depth": cls.RESEARCH_DEPTH,
            "max_sources": cls.MAX_SOURCES,
            "paper_target_length": cls.PAPER_TARGET_LENGTH,
            "enable_citations": cls.ENABLE_CITATIONS,
            "enable_plagiarism_check": cls.ENABLE_PLAGIARISM_CHECK
        }
    
    @classmethod
    def get_llm_config(cls) -> Dict[str, Any]:
        """Get LLM-specific configuration"""
        return {
            "model_name": cls.GROQ_MODEL,
            "max_tokens": cls.MAX_TOKENS,
            "temperature": cls.TEMPERATURE
        }
    
    @classmethod
    def get_ui_config(cls) -> Dict[str, Any]:
        """Get UI-specific configuration"""
        return {
            "port": cls.STREAMLIT_PORT,
            "host": cls.STREAMLIT_HOST,
            "enable_debug": cls.ENABLE_DEBUG
        }
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("🔧 Agentic Research Assistant Configuration")
        print("=" * 50)
        
        print(f"📡 API Keys:")
        print(f"  GROQ: {'✅ Set' if cls.GROQ_API_KEY else '❌ Missing'}")
        print(f"  Tavily: {'✅ Set' if cls.TAVILY_API_KEY else '❌ Missing'}")
        
        print(f"\n🔬 Research Settings:")
        print(f"  Depth: {cls.RESEARCH_DEPTH}")
        print(f"  Max Sources: {cls.MAX_SOURCES}")
        print(f"  Target Length: {cls.PAPER_TARGET_LENGTH:,} characters")
        
        print(f"\n🤖 Agent Features:")
        print(f"  Citations: {'✅ Enabled' if cls.ENABLE_CITATIONS else '❌ Disabled'}")
        print(f"  Plagiarism Check: {'✅ Enabled' if cls.ENABLE_PLAGIARISM_CHECK else '❌ Disabled'}")
        
        print(f"\n🌐 UI Settings:")
        print(f"  Port: {cls.STREAMLIT_PORT}")
        print(f"  Host: {cls.STREAMLIT_HOST}")
        print(f"  Debug Mode: {'✅ Enabled' if cls.ENABLE_DEBUG else '❌ Disabled'}")
        
        print(f"\n⚡ Performance:")
        print(f"  Caching: {'✅ Enabled' if cls.ENABLE_CACHING else '❌ Disabled'}")
        print(f"  Max Concurrent: {cls.MAX_CONCURRENT_REQUESTS}")
        
        # Validate configuration
        issues = cls.validate()
        if issues:
            print(f"\n⚠️  Configuration Issues:")
            for key, issue in issues.items():
                print(f"  {key}: {issue}")
        else:
            print(f"\n✅ Configuration is valid!")

# Default research configurations for different types
RESEARCH_CONFIGS = {
    "quick_review": {
        "max_sources": 5,
        "paper_target_length": 2000,
        "research_depth": "quick"
    },
    "focused": {
        "max_sources": 10,
        "paper_target_length": 4000,
        "research_depth": "focused"
    },
    "comprehensive": {
        "max_sources": 15,
        "paper_target_length": 8000,
        "research_depth": "comprehensive"
    },
    "deep_analysis": {
        "max_sources": 20,
        "paper_target_length": 12000,
        "research_depth": "deep"
    }
}

# Agent-specific configurations
AGENT_CONFIGS = {
    "literature_agent": {
        "search_timeout": 30,
        "min_relevance_score": 7.0,
        "enable_abstract_extraction": True
    },
    "data_agent": {
        "max_data_points": 10,
        "confidence_threshold": 0.7,
        "enable_trend_analysis": True
    },
    "writer_agent": {
        "enable_outline_generation": True,
        "enable_citation_integration": True,
        "writing_style": "academic"
    },
    "editor_agent": {
        "enable_grammar_check": True,
        "enable_style_check": True,
        "enable_formatting": True
    }
}

if __name__ == "__main__":
    # Print configuration when run directly
    Config.print_config()
