@echo off
setlocal

echo.
echo  CDMaker  ^>  http://localhost:5000
echo.

start "" http://localhost:5000
python app.py

pause
