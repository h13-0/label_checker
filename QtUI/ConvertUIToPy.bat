call %USERPROFILE%\anaconda3\Scripts\activate.bat %USERPROFILE%\anaconda3
call conda activate label_checker
python -m PyQt6.uic.pyuic LabelChecker.ui -o Ui_LabelChecker.py
python -m PyQt6.uic.pyuic TemplateEditor.ui -o Ui_TemplateEditor.py