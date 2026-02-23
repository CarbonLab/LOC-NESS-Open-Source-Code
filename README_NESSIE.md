# NESSIE - LOCNESS Data Visualization Dashboard

A Dash-based web application for visualizing oceanographic data from the LOCNESS project.

## Deployment on Render

This app is configured for deployment on Render.com. The following files are required:

- `Nessie.py` - Main application file
- `requirements.txt` - Python dependencies
- `Procfile` - Tells Render how to start the app
- `runtime.txt` - Specifies Python version
- `render.yaml` - Render deployment configuration

## Local Development

1. Create a virtual environment:
   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   python Nessie.py
   ```

## Deployment Issues & Solutions

### Common Problems:

1. **Memory Issues**: The app loads large datasets. If you encounter memory errors:
   - Use the `starter` plan on Render (1GB RAM)
   - Data is now loaded lazily to reduce startup memory usage

2. **Data Loading Failures**: If external data sources are unavailable:
   - The app will show "No data available" messages
   - Check the Render logs for specific error messages

3. **Build Failures**: If the build fails:
   - Ensure all dependencies in `requirements.txt` are compatible
   - Check that Python 3.12 is specified in `runtime.txt`

### Health Check:

The app includes a health check endpoint at `/health` that returns:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00"
}
```

### Environment Variables:

- `PORT`: Automatically set by Render
- `PYTHON_VERSION`: Set to 3.12.0
- `PYTHONUNBUFFERED`: Set to 1 for better logging

## Data Sources

The app fetches data from:
- MBARI GliderViz data repository
- Various oceanographic data sources

## Troubleshooting

1. Check Render build logs for specific error messages
2. Verify all required files are committed to your repository
3. Ensure the `Procfile` points to `Nessie:server`
4. Check that `gunicorn` is in your requirements.txt

## Support

For deployment issues, check:
1. Render build logs
2. Application logs in Render dashboard
3. Health check endpoint response
