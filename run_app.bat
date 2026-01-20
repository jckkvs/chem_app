@echo off
echo Starting Chemical ML Platform...

:: Start Django Server
start "Django API" cmd /k "python manage.py runserver"

:: Start Huey Consumer
start "Huey Worker" cmd /k "huey_consumer.py core.tasks.huey"

:: Wait a bit for API to come up
timeout /t 5

:: Start Streamlit
start "Frontend" cmd /k "streamlit run frontend_streamlit/app.py"

echo All services started.
pause
