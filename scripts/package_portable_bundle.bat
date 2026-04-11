@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0\.."

set "PI="
if exist ".venv\Scripts\pyinstaller.exe" set "PI=.venv\Scripts\pyinstaller.exe"
if not defined PI set "PI=pyinstaller"

"%PI%" --noconfirm graph_traffic_dataset_creator.spec
if errorlevel 1 exit /b 1

set "BUNDLE=dist\portable-graph-traffic-dataset-creator"
if exist "%BUNDLE%" rmdir /s /q "%BUNDLE%"
mkdir "%BUNDLE%" || exit /b 1
mkdir "%BUNDLE%\sumo" || exit /b 1

copy /y "dist\graph-traffic-dataset-creator.exe" "%BUNDLE%\" || exit /b 1

set "SUMO_SRC="
if defined PORTABLE_BUNDLE_SUMO_HOME set "SUMO_SRC=!PORTABLE_BUNDLE_SUMO_HOME!"
if not defined SUMO_SRC if defined SUMO_HOME set "SUMO_SRC=!SUMO_HOME!"

if defined SUMO_SRC (
  if exist "!SUMO_SRC!\bin\sumo.exe" if exist "!SUMO_SRC!\tools" (
    echo Copying SUMO from: !SUMO_SRC!
    xcopy /e /i /y "!SUMO_SRC!\*" "%BUNDLE%\sumo\"
    goto :done
  )
)

echo No valid SUMO_HOME for copy; see dist\portable-graph-traffic-dataset-creator\sumo\README.txt
> "%BUNDLE%\sumo\README.txt" (
  echo Put a full SUMO install here: bin\sumo.exe and tools\
  echo https://sumo.dlr.de/docs/Downloads.php
)
:done
echo Portable bundle: %BUNDLE%
