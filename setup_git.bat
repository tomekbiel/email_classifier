@echo off
cd /d "C:\python\email_classifier"
echo Current directory: %CD%
git init
git add .
git commit -m "Initial commit - Email Classification Pipeline"
git remote add origin https://github.com/tomekbiel/email_classifier.git
git branch -M main
git push -u origin main
pause
