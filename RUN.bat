@echo off
REM Launch RocketSimVis without a terminal window.
REM pythonw.exe = no console; start "" = bat exits immediately after spawning.
start "" pythonw "%~dp0src\main.py"
