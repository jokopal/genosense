# Genosense - Palm Oil Monitoring System

A comprehensive web-based platform for monitoring and analyzing Ganoderma boninense infections in palm oil plantations using cutting-edge machine learning techniques.

## Features

- **Interactive Map**: Visualize infection data geographically using Leaflet.js
- **ML-based Detection**: Automatically identify G. boninense infections from uploaded multispectral imagery
- **Infection Spread Prediction**: Forecast the spread of infections using Cellular Automata modeling
- **Trend Analysis**: Visualize infection trends over time with interactive charts
- **Pattern Recognition**: Analyze spatial patterns of infection spread
- **Cloud Integration**: Support for model hosting on Google Cloud Storage
- **Multiple Model Support**: Compatible with both Keras (.h5) and PyTorch (.pt) models

## Tech Stack

- **Backend**: Flask (Python)
- **Database**: PostgreSQL
- **ML Frameworks**: TensorFlow/Keras, PyTorch
- **Frontend**: JavaScript, HTML, CSS
- **Map Visualization**: Leaflet.js
- **Charts**: Chart.js
- **Deployment**: Google App Engine, CI/CD with GitHub Actions

## Installation

1. Clone the repository
   ```
   git clone https://github.com/yourusername/genosense.git
   cd genosense
   ```

2. Set up virtual environment (recommended)
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```
   pip install -r requirements-prod.txt
   ```

4. Set up environment variables
   ```
   export FLASK_ENV=development
   export DATABASE_URL=postgresql://username:password@localhost:5432/genosense
   ```

5. Run the application
   ```
   python main.py
   ```

## Project Structure

- `/static` - Frontend assets (CSS, JS, images)
- `/templates` - HTML templates
- `/models` - ML model definitions and processing logic
- `/uploads` - User-uploaded images
- `/tests` - Unit and integration tests

## API Endpoints

- `/api/infection_data` - Get infection data for map visualization
- `/api/trend_data` - Get trend data for charts
- `/api/predict` - Generate infection spread predictions
- `/api/model_info` - Get information about available ML models
- `/api/models` - Get list of all available models (local and GCP)
- `/api/gcp/configure` - Configure Google Cloud Platform credentials
- `/api/process_image_with_model` - Process an image with a specific model
- `/api/upload_model` - Upload a new ML model file

## Deployment

The application is configured for deployment to Google App Engine using GitHub Actions CI/CD:

1. Set up the required GitHub Secrets:
   - `GCP_PROJECT_ID` - Your Google Cloud Project ID
   - `GCP_SA_KEY` - Service Account Key JSON for authentication
   - `GCP_MODEL_BUCKET` - GCS bucket name for model storage

2. Push to the `main` branch to trigger deployment, or manually run the workflow.

## Testing

Run tests using pytest:
```
python -m pytest tests/
```

## License

MIT