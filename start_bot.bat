@echo off
title TradingBot [4H] - validated config
cd /d C:\Users\seemonster\trading-bot
echo ==================================================
echo   TRADING BOT [4H] - validated config
echo   Folder: C:\Users\seemonster\trading-bot
echo ==================================================
:loop
python bot.py
if %ERRORLEVEL% EQU 1 (
    echo Another instance is already running. Exiting.
    pause
    exit /b 0
)
echo Bot stopped, restarting in 10 seconds...
timeout /t 10
goto loop
