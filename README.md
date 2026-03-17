# Bangalore Home Price Prediction
 
A machine learning web application that predicts residential property prices in Bangalore, India. Users can input property details — area, number of bedrooms, bathrooms, and location — and get an estimated price in real time.
 
Built with **Python**, **scikit-learn**, and **Streamlit**.
 
## Demo
 
<!-- Replace with your deployed URL or a screenshot -->
<!-- ![App Screenshot](screenshot.png) -->
 
Run locally:
 
```bash
streamlit run app.py
```
 
## Features
 
- Predicts home prices across **241 locations** in Bangalore
- Inputs: square footage, BHK (bedrooms), bathrooms, and location
- Displays price in **Lakhs** or **Crores** (Indian currency)
- Instant predictions powered by a pre-trained Linear Regression model
 
## Project Structure
 
```
├── app.py                      # Streamlit web application (main entry point)
├── requirements.txt            # Python dependencies
├── server/
│   ├── artifacts/
│   │   ├── banglore_home_prices_model.pickle        # Trained ML model
│   │   └── banglore_home_prices_model_columns.json  # Feature column definitions
│   ├── server.py               # Flask backend (alternative to Streamlit)
│   └── util.py                 # Model loading and prediction utilities
├── client/
│   ├── app.html                # Frontend UI (used with Flask server)
│   ├── app.js                  # Frontend logic
│   └── app.css                 # Styling
└── model/
    ├── Real_State.ipynb        # Jupyter notebook — full data analysis and model training
    ├── retrain.py              # Script to retrain the model
    └── Bengaluru_House_Data.csv # Dataset (13,320 records)
```
 
## How It Works
 
### 1. Data Collection
 
The model is trained on the [Bengaluru House Price Dataset](https://www.kaggle.com/datasets/pradeepsapparapu/bengaluru-house-datacsv) containing 13,320 property records with features like area type, location, size, total square footage, number of bathrooms, and price.
 
### 2. Data Cleaning & Feature Engineering
 
- Dropped non-essential columns: `area_type`, `availability`, `society`, `balcony`
- Extracted **BHK** (number of bedrooms) from the `size` column
- Converted `total_sqft` ranges (e.g., `1200-1500`) to their average
- Grouped rare locations (fewer than 10 data points) into an `other` category to reduce dimensionality from 1,293 to 242 locations
 
### 3. Outlier Removal
 
Three rounds of outlier removal:
 
1. **Square footage per BHK** — removed properties with less than 300 sqft per bedroom
2. **Price per sqft** — removed properties outside 1 standard deviation of the mean price per sqft for each location
3. **BHK price anomalies** — removed cases where a lower-BHK property costs more per sqft than the next higher BHK in the same location
4. **Bathroom count** — removed properties where bathrooms exceed BHK + 2
 
### 4. Model Training
 
- Applied **one-hot encoding** on locations (dropped `other` to avoid the dummy variable trap)
- Trained a **Linear Regression** model using an 80/20 train-test split
- Achieved an **R² score of ~0.845**
- Cross-validated with ShuffleSplit (5 folds) — consistent scores around 0.82–0.85
- Compared against Lasso Regression and Decision Tree Regressor using GridSearchCV — Linear Regression performed best
 
### 5. Prediction
 
The model takes 4 inputs and returns an estimated price in Lakhs:
 
| Input         | Description                        |
|---------------|------------------------------------|
| Total Sqft    | Total area in square feet          |
| BHK           | Number of bedrooms (1–5)           |
| Bathrooms     | Number of bathrooms (1–5)          |
| Location      | One of 241 Bangalore neighborhoods |
 
## Getting Started
 
### Prerequisites
 
- Python 3.8+
 
### Installation
 
```bash
# Clone the repository
git clone https://github.com/yourusername/Real-Estate-Price-Prediction.git
cd Real-Estate-Price-Prediction
 
# Install dependencies
pip install -r requirements.txt
 
# Run the app
streamlit run app.py
```
 
The app will open at `http://localhost:8501`.
 
### Retrain the Model (Optional)
 
If you want to retrain the model with updated data or a different scikit-learn version:
 
```bash
cd model
python retrain.py
```
 
This regenerates the `.pickle` and `.json` artifacts in `server/artifacts/`.
 
## Deployment
 
### Streamlit Community Cloud (Recommended)
 
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo and set `app.py` as the main file
4. Deploy — you'll get a free `*.streamlit.app` URL
 
### Other Options
 
- **Render** — create a Web Service with start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
- **Railway** — import from GitHub, same start command as Render
- **Hugging Face Spaces** — create a new Space with Streamlit SDK and upload files
 
## Tech Stack
 
- **Python** — core language
- **pandas / NumPy** — data manipulation and preprocessing
- **scikit-learn** — model training (Linear Regression)
- **Streamlit** — web UI and deployment
- **Jupyter Notebook** — exploratory data analysis
 
## Dataset
 
Source: [Bengaluru House Price Data — Kaggle](https://www.kaggle.com/datasets/pradeepsapparapu/bengaluru-house-datacsv)
 
- **Records:** 13,320
- **Final training samples:** ~7,251 (after cleaning and outlier removal)
- **Features used:** total_sqft, bath, BHK, location (one-hot encoded)
- **Target:** price (in Lakhs)
 
## License
 
This project is open source and available under the [MIT License](LICENSE).
