# Setting up GitHub Secrets for Heroku Deployment

This guide will walk you through the process of setting up GitHub Secrets for automated deployment to Heroku using GitHub Actions.

## Prerequisites

1. A GitHub account with access to the repository
2. A Heroku account with API access
3. The Heroku CLI installed locally (for getting your API key)

## Steps to Set Up GitHub Secrets

1. **Get your Heroku API Key**

   ```bash
   heroku auth:token
   ```

   This will display your Heroku API key. Copy this value.

2. **Navigate to your GitHub Repository Settings**

   Go to your GitHub repository, click on "Settings" > "Secrets and variables" > "Actions".

3. **Add the following repository secrets:**

   a. **HEROKU_API_KEY**
      - Name: `HEROKU_API_KEY`
      - Value: The API key you obtained in step 1

   b. **HEROKU_APP_NAME**
      - Name: `HEROKU_APP_NAME`
      - Value: The name of your Heroku application (e.g., "genosense")

   c. **HEROKU_EMAIL**
      - Name: `HEROKU_EMAIL`
      - Value: The email address associated with your Heroku account

4. **Add OAuth Credentials (Optional)**

   If your application uses Heroku OAuth for API access, also add these secrets:

   a. **HEROKU_OAUTH_ID**
      - Name: `HEROKU_OAUTH_ID`
      - Value: `ad99376f-c110-4e3f-bb76-548a85727ff1`

   b. **HEROKU_OAUTH_SECRET**
      - Name: `HEROKU_OAUTH_SECRET` 
      - Value: `3b3ec407-649a-4e04-b190-9afa6d6f8756`

## Verify the Configuration

Once you've added all the secrets, GitHub Actions will use these secrets to deploy your application to Heroku whenever you push to the main or master branch.

You can verify the deployment process by:

1. Making a small change to your application
2. Committing and pushing the change to your main/master branch
3. Going to the "Actions" tab in your GitHub repository to see the deployment workflow running

## Troubleshooting

If the deployment fails, check the following:

1. Verify that all secrets are correctly set up
2. Ensure your Heroku application exists
3. Check that the Heroku API key is valid and not expired
4. Verify that the email address matches your Heroku account

For more information, refer to the [GitHub Actions documentation](https://docs.github.com/en/actions) and [Heroku API documentation](https://devcenter.heroku.com/categories/platform-api).