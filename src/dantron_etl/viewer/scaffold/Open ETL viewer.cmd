@echo off
rem Double-click this file to open the ETL viewer for this project in your browser.
rem Keep the window open while you use the viewer. Close it, or press ctrl-c, to stop the viewer.
cd /d "%~dp0"
where @@command@@ >nul 2>nul
if errorlevel 1 goto fallback
@@command@@ view
goto end
:fallback
@@fallback@@
:end
pause
