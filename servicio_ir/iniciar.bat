@echo off
cd /d "%~dp0.."
call .venv\Scripts\activate.bat
python -m uvicorn servicio_ir.main:app --host 0.0.0.0 --port 8000
