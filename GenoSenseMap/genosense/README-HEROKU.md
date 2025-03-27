# Deploying Genosense to Heroku

This guide will walk you through the process of deploying the Genosense palm oil monitoring system to Heroku.

## Prerequisites

1. A Heroku account (sign up at https://signup.heroku.com/)
2. Heroku CLI installed (https://devcenter.heroku.com/articles/heroku-cli)
3. Git installed (https://git-scm.com/downloads)

## Deployment Steps

1. **Login to Heroku CLI**

   ```bash
   heroku login
   ```

2. **Create a new Heroku app**

   ```bash
   heroku create genosense
   ```

   This will create a new Heroku app named "genosense". If this name is already taken, choose a different name.

3. **Add the PostgreSQL add-on**

   ```bash
   heroku addons:create heroku-postgresql:hobby-dev
   ```

   This will add a free PostgreSQL database to your app.

4. **Set environment variables**

   ```bash
   heroku config:set SESSION_SECRET=$(openssl rand -hex 32)
   ```

   This sets a secure random secret key for your app's session management.

5. **Deploy the application**

   ```bash
   git push heroku master
   ```

   Or if you're not on the master branch:

   ```bash
   git push heroku yourbranchname:master
   ```

6. **Run database migrations**

   ```bash
   heroku run python -c "from app import app, db; app.app_context().push(); db.create_all()"
   ```

   This will create the database tables required by the application.

7. **Open the application**

   ```bash
   heroku open
   ```

   This will open your deployed application in a browser.

## Additional Configuration

### Adding Machine Learning Models

Due to GitHub and Heroku file size limitations, ML models are not included in the repository. You have two options:

1. **Upload models through the web interface**
   - Navigate to the "/upload_model" page in your application
   - Upload your .h5 (Keras) or .pt (PyTorch) model files

2. **Configure Google Cloud Storage**
   - Navigate to the "/configure_gcp" page in your application
   - Provide your GCP credentials
   - Your models must be stored in a GCS bucket

### Scaling Your Application

If you need more resources for your app, you can upgrade your Heroku dyno:

```bash
heroku ps:scale web=1:standard-1x
```

For more information on Heroku dyno types and pricing, visit: https://www.heroku.com/pricing

### Monitoring and Logging

View application logs:

```bash
heroku logs --tail
```

Visit the Heroku dashboard for detailed monitoring and logging capabilities.

## Troubleshooting

If your application fails to start, check the logs:

```bash
heroku logs --tail
```

Common issues:
- Database connection errors: Check your DATABASE_URL configuration
- Dependencies missing: Ensure all requirements are listed in requirements.txt
- Port binding issues: Make sure your app is binding to the PORT environment variable

For more help, refer to the Heroku documentation: https://devcenter.heroku.com/articles/getting-started-with-python