#!/usr/bin/env python3
"""
Script to build SemantiCore documentation.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(command, cwd=None):
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✅ {command}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {command}")
        print(f"Error: {e.stderr}")
        return None

def install_dependencies():
    """Install documentation dependencies."""
    print("📦 Installing documentation dependencies...")
    
    dependencies = [
        "sphinx",
        "sphinx-rtd-theme",
        "sphinx-copybutton",
        "sphinx-tabs",
        "myst-parser",
        "sphinxcontrib-spelling",
        "doc8",
        "sphinx-lint"
    ]
    
    for dep in dependencies:
        run_command(f"pip install {dep}")

def build_documentation():
    """Build the documentation."""
    print("🔨 Building documentation...")
    
    docs_dir = Path("docs")
    if not docs_dir.exists():
        print("❌ docs directory not found!")
        return False
    
    # Change to docs directory
    os.chdir(docs_dir)
    
    # Clean previous builds
    if Path("_build").exists():
        shutil.rmtree("_build")
    
    # Build HTML documentation
    result = run_command("make html")
    if result is None:
        return False
    
    # Check for broken links
    print("🔗 Checking for broken links...")
    result = run_command("make linkcheck")
    if result is None:
        print("⚠️  Link check failed, but continuing...")
    
    # Run doctests
    print("🧪 Running doctests...")
    result = run_command("make doctest")
    if result is None:
        print("⚠️  Doctests failed, but continuing...")
    
    return True

def serve_documentation():
    """Serve the documentation locally."""
    print("🌐 Starting local documentation server...")
    
    docs_dir = Path("docs")
    build_dir = docs_dir / "_build" / "html"
    
    if not build_dir.exists():
        print("❌ Documentation not built. Run build first.")
        return False
    
    os.chdir(build_dir)
    
    print("📖 Documentation available at: http://localhost:8000")
    print("Press Ctrl+C to stop the server.")
    
    try:
        run_command("python -m http.server 8000")
    except KeyboardInterrupt:
        print("\n🛑 Server stopped.")
    
    return True

def deploy_documentation():
    """Deploy documentation to GitHub Pages."""
    print("🚀 Deploying documentation to GitHub Pages...")
    
    # Check if we're on the main branch
    result = run_command("git branch --show-current")
    if result and "main" not in result.strip():
        print("❌ Not on main branch. Deployment only works from main.")
        return False
    
    # Build documentation
    if not build_documentation():
        return False
    
    # Deploy to gh-pages branch
    docs_dir = Path("docs")
    build_dir = docs_dir / "_build" / "html"
    
    if not build_dir.exists():
        print("❌ Documentation not built.")
        return False
    
    # Create gh-pages branch if it doesn't exist
    run_command("git checkout --orphan gh-pages || git checkout gh-pages")
    
    # Copy built documentation
    for item in build_dir.iterdir():
        if item.is_file():
            shutil.copy2(item, ".")
        elif item.is_dir():
            shutil.copytree(item, item.name, dirs_exist_ok=True)
    
    # Commit and push
    run_command("git add .")
    run_command('git commit -m "Update documentation"')
    run_command("git push origin gh-pages")
    
    # Switch back to main branch
    run_command("git checkout main")
    
    print("✅ Documentation deployed successfully!")
    return True

def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python build_docs.py [install|build|serve|deploy|all]")
        print("\nCommands:")
        print("  install  - Install documentation dependencies")
        print("  build    - Build documentation")
        print("  serve    - Serve documentation locally")
        print("  deploy   - Deploy to GitHub Pages")
        print("  all      - Install, build, and serve")
        return
    
    command = sys.argv[1].lower()
    
    if command == "install":
        install_dependencies()
    elif command == "build":
        build_documentation()
    elif command == "serve":
        serve_documentation()
    elif command == "deploy":
        deploy_documentation()
    elif command == "all":
        install_dependencies()
        if build_documentation():
            serve_documentation()
    else:
        print(f"❌ Unknown command: {command}")

if __name__ == "__main__":
    main() 