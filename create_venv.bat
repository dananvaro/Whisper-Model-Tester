@echo off
REM Create a local virtual environment in .venv and install requirements
python -m venv .venv
echo Upgrading pip...
.venv\Scripts\python -m pip install --upgrade pip
echo Installing requirements from requirements.txt...
.venv\Scripts\python -m pip install -r requirements.txt
echo Done. To activate the environment, run:
echo    .venv\Scripts\activate