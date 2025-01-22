@echo off
mkdir yield
for %%i in (*yield*.csv) do (
    move "%%i" .\yield
)
