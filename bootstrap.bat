@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

where conda >nul 2>nul
if errorlevel 1 (
  echo [bootstrap] conda was not found in PATH.
  exit /b 1
)

set "ENV_NAME="
set "PYTHON_VERSION="

for /f "tokens=1,* delims=:" %%A in ('findstr /B /R /C:"  environment_name:" config\config.yaml') do (
  set "ENV_NAME=%%B"
)
for /f "tokens=1,* delims=:" %%A in ('findstr /B /R /C:"  python_version:" config\config.yaml') do (
  set "PYTHON_VERSION=%%B"
)

set "ENV_NAME=!ENV_NAME: =!"
set "ENV_NAME=!ENV_NAME:"=!"
set "PYTHON_VERSION=!PYTHON_VERSION: =!"
set "PYTHON_VERSION=!PYTHON_VERSION:"=!"

if "!ENV_NAME!"=="" (
  echo [bootstrap] Failed to read project.environment_name from config\config.yaml.
  exit /b 1
)

if "!PYTHON_VERSION!"=="" (
  echo [bootstrap] Failed to read project.python_version from config\config.yaml.
  exit /b 1
)

conda env list | findstr /R /C:"^[* ]*!ENV_NAME! " >nul
if errorlevel 1 (
  echo [bootstrap] Creating conda env !ENV_NAME! with Python !PYTHON_VERSION!.
  call conda create -n "!ENV_NAME!" "python=!PYTHON_VERSION!" -y
  if errorlevel 1 exit /b 1
) else (
  echo [bootstrap] Using existing conda env !ENV_NAME!.
)

echo [bootstrap] Running init.py inside conda env !ENV_NAME!.
call conda run -n "!ENV_NAME!" python init.py %*
exit /b %errorlevel%
