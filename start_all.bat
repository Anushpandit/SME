@echo off
cd /d c:\Users\Abhay\Desktop\HACKATHON\doc

REM Start Streamlit in a new window
start "Streamlit App" cmd /k "call .venv\Scripts\activate.bat && streamlit run app.py"

REM Wait 2 seconds for Streamlit to initialize
timeout /t 2 /nobreak

REM Start ngrok in another new window
start "ngrok Tunnel" cmd /k "cd /d c:\Users\Abhay\Desktop\HACKATHON\doc && .\ngrok http 8501"

echo.
echo ========================================
echo Streamlit and ngrok are starting!
echo ========================================
echo Streamlit: http://localhost:8501
echo ngrok will display your public URL
echo ========================================
echo.
pause
