@echo off
echo Multi-Modal Breast Cancer Detection - GitHub Push Script
echo ========================================================
echo.
echo Repository: https://github.com/IH-Arik/Multi-Modal-Breast-Cancer-Detection
echo.
echo If Git is not installed, please download from: https://git-scm.com/download/win
echo.
pause

echo Initializing Git repository...
git init
echo.

echo Adding all files...
git add .
echo.

echo Creating initial commit...
git commit -m "Initial commit: Multi-Modal Breast Cancer Detection Research Framework"
echo.

echo Adding remote origin...
git remote add origin https://github.com/IH-Arik/Multi-Modal-Breast-Cancer-Detection.git
echo.

echo Setting main branch...
git branch -M main
echo.

echo Pushing to GitHub...
git push -u origin main
echo.

echo ========================================================
echo If successful, visit: https://github.com/IH-Arik/Multi-Modal-Breast-Cancer-Detection
echo ========================================================
pause
