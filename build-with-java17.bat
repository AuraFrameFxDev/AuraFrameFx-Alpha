@echo off
setlocal

REM Set the Java home to the correct JDK 17 installation
set JAVA_HOME=C:\Program Files\Eclipse Adoptium\jdk-17.0.15.6-hotspot

REM Add Java bin to PATH
set PATH=%JAVA_HOME%\bin;%PATH%

REM Run Gradle with the specified command
call gradlew.bat %*

endlocal
