#!/usr/bin/env python3
"""
Git initialization script for Multi-Modal Breast Cancer Detection

This script helps initialize the Git repository and provides instructions
for pushing to GitHub.

Usage:
    python scripts/init_git.py
"""

import os
import subprocess
from pathlib import Path


def run_command(command, cwd=None):
    """Run a shell command and return the output."""
    try:
        result = subprocess.run(command, shell=True, cwd=cwd, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)


def check_git_installed():
    """Check if Git is installed."""
    success, _, _ = run_command("git --version")
    return success


def init_git_repo():
    """Initialize Git repository."""
    project_root = Path(__file__).parent.parent
    
    print("Initializing Git repository...")
    
    # Check if Git is installed
    if not check_git_installed():
        print("Git is not installed or not in PATH")
        print("Please install Git first: https://git-scm.com/downloads")
        return False
    
    # Initialize repository
    success, output, error = run_command("git init", cwd=project_root)
    if success:
        print("Git repository initialized")
    else:
        print(f"Failed to initialize Git: {error}")
        return False
    
    # Add all files
    print("Adding files to Git...")
    success, output, error = run_command("git add .", cwd=project_root)
    if success:
        print("Files added to Git")
    else:
        print(f"Failed to add files: {error}")
        return False
    
    # Initial commit
    print("Creating initial commit...")
    success, output, error = run_command('git commit -m "Initial commit: Multi-Modal Breast Cancer Detection Research Framework"', cwd=project_root)
    if success:
        print("Initial commit created")
    else:
        print(f"Failed to create commit: {error}")
        return False
    
    return True


def print_github_instructions():
    """Print instructions for GitHub setup."""
    print("\n" + "="*80)
    print("GITHUB SETUP INSTRUCTIONS")
    print("="*80)
    
    print("\n1. Create GitHub Repository:")
    print("   • Go to: https://github.com/new")
    print("   • Repository name: Multi-Modal-Breast-Cancer-Detection")
    print("   • Description: Advanced deep learning for multi-modal breast cancer detection")
    print("   • Make it Public (for research visibility)")
    print("   • DO NOT initialize with README, .gitignore, or license")
    print("   • Click 'Create repository'")
    
    print("\n2. Push to GitHub:")
    print("   • Copy the repository URL (e.g., https://github.com/IH-Arik/Multi-Modal-Breast-Cancer-Detection.git)")
    print("   • Run these commands:")
    print("     git remote add origin https://github.com/IH-Arik/Multi-Modal-Breast-Cancer-Detection.git")
    print("     git branch -M main")
    print("     git push -u origin main")
    
    print("\n3. Repository Features:")
    print("   • GitHub will automatically render your README.md")
    print("   • Your code is now professionally structured")
    print("   • Research is protected with proper licensing")
    print("   • Easy collaboration and citation")
    
    print("\n4. Next Steps:")
    print("   • Update dataset paths in configs/config.yaml")
    print("   • Run: python scripts/train_all_modalities.py")
    print("   • Share with collaborators and cite in papers")


def main():
    """Main function."""
    print("Multi-Modal Breast Cancer Detection - Git Setup")
    print("="*60)
    
    # Initialize Git repository
    if init_git_repo():
        print_github_instructions()
    else:
        print("\nGit initialization failed. Please check the error messages above.")
        print("You can also manually initialize Git using:")
        print("   git init")
        print("   git add .")
        print("   git commit -m 'Initial commit'")


if __name__ == "__main__":
    main()
