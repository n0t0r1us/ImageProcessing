@echo off
setlocal enabledelayedexpansion
set i=1
for %%f in (image_*.jpg) do (
    ren "%%f" "sunflower_!i!.jpg"
    set /a i+=1
)
