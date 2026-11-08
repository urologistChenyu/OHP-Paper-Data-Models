@echo off
setlocal
set "TASK_PYTHON=%~dp0..\02_End_to_End\runtime\python312\python.exe"
if not exist "%TASK_PYTHON%" (
  echo Keep 01_Paper_Data_Models and 02_End_to_End beside each other.
  pause
  exit /b 1
)
"%TASK_PYTHON%" -B "%~dp0analysis\reproduce_prediction_results.py" --input "%~dp0data\predictions\all_cohorts_predictions.csv" --output-dir "%~dp0reproduced_results"
set "TASK_EXIT=%ERRORLEVEL%"
pause
exit /b %TASK_EXIT%
