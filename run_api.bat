@echo off
echo ======================================================================
echo 🫁 Starting ChestAI Radiology REST API (FastAPI)...
echo ======================================================================
echo Swagger Documentation will be available at: http://localhost:8000/docs
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
pause
