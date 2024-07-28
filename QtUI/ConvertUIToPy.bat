call %USERPROFILE%\anaconda3\Scripts\activate.bat %USERPROFILE%\anaconda3
call conda activate label_checker
python -m PyQt6.uic.pyuic mainwindow.ui -o UI.py