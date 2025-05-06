```python
# start vir env
python -m venv test_env
source test_env/bin/activate

# install dependencies
pip install -r requirements.txt

# run the app
python3.11 preprocessing.py
python3.11 vectorize.py
python3.11 query.py
```
