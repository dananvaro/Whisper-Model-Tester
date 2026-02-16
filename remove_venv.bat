@echo off
REM Remove the local .venv directory
if exist .venv (
    rmdir /s /q .venv
    echo Removed .venv
) else (
    echo No .venv directory found
)