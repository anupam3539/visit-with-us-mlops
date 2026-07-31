# Visit with Us — Wellness Tourism MLOps

End-to-end pipeline for validating customer data, preparing stratified train/test artifacts, tuning and tracking a Random Forest model, committing the best model, and serving predictions with Streamlit.

## Local run
```bash
pip install -r tourism_project/requirements.txt
python tourism_project/model_building/data_register.py
python tourism_project/model_building/prep.py
python tourism_project/model_building/train.py
pip install -r tourism_project/deployment/requirements.txt
streamlit run tourism_project/deployment/app.py
```

## Streamlit Community Cloud
Connect this repository and set the main file to `tourism_project/deployment/app.py`. The trained model is committed automatically by GitHub Actions.
