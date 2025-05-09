import requests

def verify_certificate(image_path):
    url = 'http://localhost:5000/verify'
    with open(image_path, 'rb') as f:
        files = {'image': f}
        response = requests.post(url, files=files)
    return response.json()

# Test with test-3.jpg
result = verify_certificate('TEST/test-3.jpg')
print("Result for test-3.jpg:", result)

# Test health endpoint
health = requests.get('http://localhost:5000/health')
print("\nAPI Health:", health.json()) 