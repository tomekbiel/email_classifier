@echo off
cd /d "C:\python\email_classifier"
echo Adding final translation changes to git...
git add .
echo Committing final translation changes...
git commit -m "Complete English translation - all Polish comments and messages removed

- Fixed all Unicode encoding errors in logging
- Translated all remaining Polish comments in pipeline.py
- Translated all remaining Polish comments in text_preprocessor.py  
- Translated all remaining Polish comments in vectorizer.py
- Translated all remaining Polish comments in data_structurer.py
- All codebase now fully in English with no Polish characters
- Pipeline runs without UnicodeEncodeError
- All logging messages in professional English"
echo Pushing to main repository...
git remote remove origin
git remote add origin https://github.com/tomekbiel/email_classifier.git
git push origin main
echo Pushing to school repository...
git remote remove school-origin
git remote add school-origin https://github.com/tomaszbielNCI/email_classifier.git
git push school-origin main
echo Complete translation pushed to both repositories!
pause
