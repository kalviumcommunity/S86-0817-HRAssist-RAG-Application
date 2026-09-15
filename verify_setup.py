#!/usr/bin/env python3
"""Setup verification script for HRAssist RAG Application.

Run this after installation to verify all components are working correctly.

Usage:
    python verify_setup.py
"""

import sys
import os
from pathlib import Path


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def check_python_version():
    """Verify Python version is 3.9+."""
    print_section("Python Version")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("[X] FAIL: Python 3.9+ required")
        return False
    print("[OK] PASS: Python version OK")
    return True


def check_dependencies():
    """Verify required packages are installed."""
    print_section("Dependencies")
    
    required = [
        "openai",
        "chromadb",
        "tiktoken",
        "fastapi",
        "streamlit",
        "pypdf",
        "beautifulsoup4",
        "dotenv",
        "pydantic",
    ]
    
    all_ok = True
    for package in required:
        try:
            if package == "dotenv":
                __import__("dotenv")
            elif package == "beautifulsoup4":
                __import__("bs4")
            else:
                __import__(package)
            print(f"[OK] {package}")
        except ImportError:
            print(f"[X] {package} -- NOT INSTALLED")
            all_ok = False
    
    if all_ok:
        print("\n[OK] PASS: All dependencies installed")
    else:
        print("\n[X] FAIL: Missing dependencies (run: pip install -r requirements.txt)")
    
    return all_ok


def check_openai_version():
    """Verify openai package is v1.x, not 0.x."""
    print_section("OpenAI SDK Version")
    
    try:
        import openai
        version = openai.__version__
        print(f"openai version: {version}")
        
        major = int(version.split(".")[0])
        if major < 1:
            print("[X] FAIL: openai 1.x required (you have 0.x)")
            print("Fix: pip install --upgrade 'openai>=1.10.0'")
            return False
        
        print("[OK] PASS: OpenAI SDK version OK")
        return True
    except Exception as e:
        print(f"[X] FAIL: Could not check version -- {e}")
        return False


def check_env_file():
    """Verify .env file exists and has required keys."""
    print_section("Environment Configuration")
    
    env_path = Path(".env")
    if not env_path.exists():
        print("[X] FAIL: .env file not found")
        print("Fix: cp .env.example .env")
        return False
    
    print("[OK] .env file exists")
    
    # Load and check keys
    from dotenv import dotenv_values
    config = dotenv_values(".env")
    
    required_keys = ["OPENAI_API_KEY", "EMBED_MODEL", "CHAT_MODEL"]
    all_ok = True
    
    for key in required_keys:
        value = config.get(key, "")
        if not value or value.startswith("your_"):
            print(f"[X] {key} -- NOT SET")
            all_ok = False
        else:
            masked = value[:8] + "..." if len(value) > 8 else "***"
            print(f"[OK] {key} = {masked}")
    
    if all_ok:
        print("\n[OK] PASS: Environment configured")
    else:
        print("\n[X] FAIL: Missing or placeholder values in .env")
    
    return all_ok


def check_project_structure():
    """Verify expected directories and files exist."""
    print_section("Project Structure")
    
    expected = [
        "src",
        "src/app.py",
        "src/rag_pipeline.py",
        "frontend",
        "frontend/app.py",
        "data",
        "tests",
        "requirements.txt",
        "README.md",
        "SETUP.md",
    ]
    
    all_ok = True
    for item in expected:
        path = Path(item)
        if path.exists():
            print(f"[OK] {item}")
        else:
            print(f"[X] {item} -- MISSING")
            all_ok = False
    
    if all_ok:
        print("\n[OK] PASS: Project structure OK")
    else:
        print("\n[X] FAIL: Missing files/directories")
    
    return all_ok


def check_imports():
    """Verify core modules can be imported."""
    print_section("Module Imports")
    
    modules = [
        "src.app",
        "src.rag_pipeline",
        "src.embedding_generator",
        "src.retriever",
        "src.guardrails",
        "src.grounded_generation",
        "src.citations",
        "src.document_loader",
    ]
    
    all_ok = True
    for module in modules:
        try:
            __import__(module)
            print(f"[OK] {module}")
        except ImportError as e:
            print(f"[X] {module} -- {e}")
            all_ok = False
        except Exception as e:
            print(f"[!] {module} -- Import succeeded but raised: {e}")
            # Don't fail on lazy-init validation errors
    
    if all_ok:
        print("\n[OK] PASS: All modules import successfully")
    else:
        print("\n[X] FAIL: Some imports failed")
    
    return all_ok


def check_chromadb():
    """Verify ChromaDB directory exists."""
    print_section("Vector Database")
    
    chroma_path = Path("chroma_db")
    if not chroma_path.exists():
        print("[!] WARNING: chroma_db/ directory not found")
        print("This is OK if you haven't indexed documents yet.")
        print("Upload a document via POST /documents to create it.")
        return True
    
    print("[OK] chroma_db/ directory exists")
    
    # Check if it has content
    db_file = chroma_path / "chroma.sqlite3"
    if db_file.exists():
        size = db_file.stat().st_size
        print(f"[OK] Database file exists ({size:,} bytes)")
    else:
        print("[!] Database file not found (no docs indexed yet)")
    
    print("\n[OK] PASS: ChromaDB directory OK")
    return True


def main():
    """Run all verification checks."""
    print("\n" + "=" * 70)
    print("  HRAssist RAG Application — Setup Verification")
    print("=" * 70)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("OpenAI SDK Version", check_openai_version),
        ("Environment Config", check_env_file),
        ("Project Structure", check_project_structure),
        ("Module Imports", check_imports),
        ("Vector Database", check_chromadb),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n[X] ERROR in {name}: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "[OK] PASS" if result else "[X] FAIL"
        print(f"{status}  {name}")
    
    print("\n" + "=" * 70)
    if passed == total:
        print("[SUCCESS] ALL CHECKS PASSED")
        print("\nYour HRAssist RAG Application is ready to run!")
        print("\nNext steps:")
        print("  1. Start backend:  uvicorn src.app:app --reload")
        print("  2. Start frontend: streamlit run frontend/app.py")
        print("  3. Open http://localhost:8501")
        print("\nSee SETUP.md for detailed instructions.")
        return 0
    else:
        print(f"[WARNING] {total - passed} CHECK(S) FAILED")
        print("\nPlease fix the issues above before running the application.")
        print("See TROUBLESHOOTING.md for solutions.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
