import secrets
import os

# Generate a secure API key
API_KEY = secrets.token_urlsafe(32)

# Save it to file
with open('.api_key', 'w') as f:
    f.write(API_KEY)

print(f"Your API key has been generated: {API_KEY}")
print("This key has been saved to .api_key file")
print("\nTo use the API, include this key in the X-API-Key header of your requests") 