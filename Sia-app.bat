@echo off
cd /d C:\Users\ahmed.amin\Desktop\SIA_App

echo.
echo Activating virtual environment...
call .\venv\Scripts\activate

echo.
echo Installing required Python packages...
pip install --quiet --disable-pip-version-check ^
    numpy==1.26.4 ^
    scipy==1.11.4 ^
    pandas ^
    statsmodels==0.14.1 ^
    pmdarima==2.0.4 ^
    xgboost ^
    lightgbm ^
    catboost ^
    prophet ^
    plotly ^
    PyQt5 ^
    openpyxl ^
    xlrd ^
    seaborn

echo.
echo Running the application...
python main.py

echo.
pause
