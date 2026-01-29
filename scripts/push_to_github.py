#!/usr/bin/env python3
"""
GitHub Push Helper Script for Multi-Modal Breast Cancer Detection

This script provides the exact commands needed to push your project to GitHub.

Repository: https://github.com/IH-Arik/Multi-Modal-Breast-Cancer-Detection
"""

import os
import subprocess
from pathlib import Path


def run_git_command(command, cwd=None):
    """Run a Git command and display the output."""
    try:
        result = subprocess.run(command, shell=True, cwd=cwd, capture_output=True, text=True)
        print(f"$ {command}")
        if result.stdout:
            print(result.stdout)
        if result.stderr and "warning:" not in result.stderr.lower():
            print(f"Error: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"Failed to run command: {e}")
        return False


def main():
    """Main function to guide GitHub setup."""
    project_root = Path(__file__).parent.parent
    
    print("="*80)
    print("GITHUB PUSH HELPER - Multi-Modal Breast Cancer Detection")
    print("="*80)
    print(f"Repository: https://github.com/IH-Arik/Multi-Modal-Breast-Cancer-Detection")
    print(f"Project Path: {project_root}")
    print("="*80)
    
    commands = [
        "git init",
        "git add .",
        'git commit -m \"Initial commit: Multi-Modal Breast Cancer Detection Research Framework\"',
        "git remote add origin https://github.com/IH-Arik/Multi-Modal-Breast-Cancer-Detection.git",
        "git branch -M main",
        "git push -u origin main"
    ]
    
    print("\nCommands to execute:")
    print("-" * 40)
    for i, cmd in enumerate(commands, 1):
        print(f"{i}. {cmd}")
    
    print("\n" + "="*80)
    print("EXECUTING COMMANDS...")
    print("="*80)
    
    success_count = 0
    for i, cmd in enumerate(commands, 1):
        print(f"\n[{i}/{len(commands)}] Executing: {cmd}")
        if run_git_command(cmd, cwd=project_root):
            success_count += 1
            print("✓ Success")
        else:
            print("✗ Failed")
    
    print("\n" + "="*80)
    print(f"COMPLETED: {success_count}/{len(commands)} commands successful")
    print("="*80)
    
    if success_count == len(commands):
        print("🎉 SUCCESS! Your project is now on GitHub!")
        print("📁 Repository: https://github.com/IH-Arik/Multi-Modal-Breast-Cancer-Detection")
        print("\nNext steps:")
        print("1. Visit your repository on GitHub")
        print("2. Update dataset paths in configs/config.yaml")
        print("3. Run: python scripts/train_all_modalities.py")
    else:
        print("⚠️  Some commands failed. Please check the output above.")
        print("You may need to:")
        print("1. Install Git from https://git-scm.com/downloads")
        print("2. Check your internet connection")
        print("3. Verify your GitHub credentials")


if __name__ == "__main__":
    main()
