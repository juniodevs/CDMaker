@echo off
setlocal

echo.
echo  CDMaker ^— Setup
echo  ================
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo  [ERRO] Python nao encontrado. Instale em https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo  Python encontrado: %PYVER%

echo.
echo  Instalando dependencias...
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo  [ERRO] Falha ao instalar dependencias.
    pause
    exit /b 1
)

echo.
echo  ================================================
echo   Setup concluido! Execute run.bat para iniciar.
echo  ================================================
echo.
pause
