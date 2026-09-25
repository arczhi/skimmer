@echo off
REM Local Focus Reader - Windows launcher (ONNX Runtime backend)
setlocal

set READER_BACKEND=onnx
if "%READER_PORT%"=="" set READER_PORT=8899

echo [reader] backend=%READER_BACKEND% port=%READER_PORT%
echo [reader] open http://127.0.0.1:%READER_PORT% after the model loads

python app.py --port %READER_PORT%
if errorlevel 1 (
  echo.
  echo [reader] failed. check:
  echo   - Python 3.10+ on PATH  ^(python --version^)
  echo   - pip install -r requirements.txt
  echo   - models_onnx\student_salience\ copied next to app.py
  pause
)
endlocal
