#!/usr/bin/env python3
"""
Agentic Launcher for the Agentic Research Assistant UI
"""

import subprocess
import sys
import os
import json
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        "streamlit",
        "plotly", 
        "pandas",
        "langchain",
        "langchain-groq",
        "langchain-community",
        "python-dotenv"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_dependencies():
    """Install missing dependencies"""
    print("📦 Installing missing dependencies...")
    
    # Try different installation methods
    installation_methods = [
        # Method 1: Try pip directly
        lambda: subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "streamlit", "plotly", "pandas", "langchain", 
            "langchain-groq", "langchain-community", "python-dotenv"
        ]),
        # Method 2: Try pip with --user flag
        lambda: subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--user",
            "streamlit", "plotly", "pandas", "langchain", 
            "langchain-groq", "langchain-community", "python-dotenv"
        ]),
        # Method 3: Try using requirements.txt
        lambda: subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ]),
        # Method 4: Try using requirements.txt with --user
        lambda: subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--user", "-r", "requirements.txt"
        ])
    ]
    
    for i, method in enumerate(installation_methods, 1):
        try:
            print(f"  Trying installation method {i}...")
            method()
            print("✅ Dependencies installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"  Method {i} failed: {e}")
            continue
        except FileNotFoundError:
            print(f"  Method {i} failed: pip not found")
            continue
    
    print("❌ All installation methods failed")
    return False

def check_environment():
    """Check environment setup"""
    print("🔍 Checking environment setup...")
    
    # Check if .env file exists
    if not os.path.exists(".env"):
        print("⚠️  Warning: .env file not found!")
        print("Please create a .env file with your API keys:")
        print("GROQ_API_KEY=your_groq_api_key_here")
        print("TAVILY_API_KEY=your_tavily_api_key_here")
        print()
        
        # Create template .env file
        env_template = """# API Keys for Agentic Research Assistant
# Get your keys from:
# GROQ: https://console.groq.com/
# TAVILY: https://tavily.com/

GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here

# Optional: Research Configuration
RESEARCH_DEPTH=comprehensive
MAX_SOURCES=15
PAPER_TARGET_LENGTH=5000
"""
        
        try:
            with open(".env", "w") as f:
                f.write(env_template)
            print("📝 Created .env template file. Please update with your actual API keys.")
        except Exception as e:
            print(f"❌ Failed to create .env template: {e}")
    
    # Check if src directory exists
    if not os.path.exists("src"):
        print("❌ Error: src directory not found!")
        print("Please ensure you're running this from the project root directory.")
        return False
    
    # Check if core module exists
    if not os.path.exists("src/core/assistant.py"):
        print("❌ Error: Core research assistant module not found!")
        print("Please ensure all source files are properly installed.")
        return False
    
    # Check if UI module exists
    if not os.path.exists("src/ui/interface.py"):
        print("❌ Error: Agentic UI module not found!")
        print("Please ensure all source files are properly installed.")
        return False
    
    print("✅ Environment setup looks good!")
    return True

def launch_ui():
    """Launch the agentic research assistant UI"""
    print("🚀 Launching Agentic Research Assistant UI...")
    print("=" * 60)
    
    # Check environment first
    if not check_environment():
        print("❌ Environment check failed. Please fix the issues above.")
        return False
    
    # Check dependencies
    missing_packages = check_dependencies()
    if missing_packages:
        print(f"📦 Missing packages: {', '.join(missing_packages)}")
        if not install_dependencies():
            print("❌ Failed to install dependencies automatically.")
            print("\n🔧 Manual Installation Options:")
            print("1. Install pip first:")
            print("   python -m ensurepip --upgrade")
            print("2. Then install dependencies:")
            print("   pip install -r requirements.txt")
            print("3. Or install packages individually:")
            print(f"   pip install {' '.join(missing_packages)}")
            print("\n💡 Alternative: Use conda if available:")
            print("   conda install streamlit plotly pandas")
            print("   pip install langchain langchain-groq langchain-community python-dotenv")
            return False
    
    # Launch Streamlit
    try:
        print("🌐 Starting Streamlit server...")
        print("📱 UI will open in your default browser")
        print("🔗 Local URL: http://localhost:8501")
        print("⏹️  Press Ctrl+C to stop the server")
        print("-" * 60)
        
        # Launch Streamlit with agentic UI
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "src/ui/interface.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false"
        ])
        
    except KeyboardInterrupt:
        print("\n🛑 Streamlit server stopped by user")
    except Exception as e:
        print(f"❌ Error launching Streamlit: {e}")
        print("Please ensure Streamlit is installed: pip install streamlit")
        return False
    
    return True

def show_project_info():
    """Display project information"""
    print("🎓 Agentic Research Assistant")
    print("=" * 60)
    print("A powerful AI-powered research automation system featuring:")
    print("• Multi-Agent Architecture (5 specialized AI agents)")
    print("• Advanced Literature Review & Analysis")
    print("• Comprehensive Data Insights")
    print("• Automated Research Paper Generation")
    print("• Quality Assurance & Editing")
    print("• Beautiful Interactive Dashboard")
    print("• Multiple Export Formats")
    print("• Research History Tracking")
    print("• Performance Metrics & Analytics")
    print("=" * 60)

def main():
    """Main launcher function"""
    show_project_info()
    
    # Check if running in correct directory
    if not os.path.exists("src"):
        print("❌ Error: Please run this script from the project root directory")
        print("Current directory:", os.getcwd())
        print("Expected structure:")
        print("  project_root/")
        print("  ├── src/")
        print("  │   ├── core/")
        print("  │   └── ui/")
        print("  ├── launch_agentic_ui.py")
        print("  └── .env")
        return
    
    # Launch UI
    success = launch_ui()
    
    if success:
        print("🎉 Agentic Research Assistant launched successfully!")
    else:
        print("❌ Failed to launch Agentic Research Assistant")
        print("Please check the error messages above and try again.")

if __name__ == "__main__":
    main()
