@echo off
echo Compilando programa MPI...

REM Configura ambiente do Visual Studio
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

REM Compila o programa
cl distribute_number-1.c /I"C:\Program Files (x86)\Microsoft SDKs\MPI\Include" /Fe:distribute_number.exe /link /LIBPATH:"C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64" msmpi.lib

if %ERRORLEVEL% EQU 0 (
    echo.
    echo Compilacao bem-sucedida!
    echo.
    echo Executando com 4 processos...
    echo.
    "C:\Program Files\Microsoft MPI\Bin\mpiexec.exe" -n 4 distribute_number.exe
) else (
    echo.
    echo Erro na compilacao!
)

pause
