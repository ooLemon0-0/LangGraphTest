@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

where conda >nul 2>nul
if errorlevel 1 (
  echo [start] conda was not found in PATH.
  exit /b 1
)

set "ENV_NAME="

for /f "tokens=1,* delims=:" %%A in ('findstr /B /R /C:"  environment_name:" config\config.yaml') do (
  set "ENV_NAME=%%B"
)

set "ENV_NAME=!ENV_NAME: =!"
set "ENV_NAME=!ENV_NAME:"=!"

if "!ENV_NAME!"=="" (
  echo [start] Failed to read project.environment_name from config\config.yaml.
  exit /b 1
)

conda env list | findstr /R /C:"^[* ]*!ENV_NAME! " >nul
if errorlevel 1 (
  echo [start] Conda env !ENV_NAME! was not found. Run bootstrap.bat first.
  exit /b 1
)

echo [start] Running start.py inside conda env !ENV_NAME!.
call conda run -n "!ENV_NAME!" python start.py %*
exit /b %errorlevel%
