# Medical Certificate Verification System

This project provides a machine learning-based system for verifying medical certificates using deep learning. It includes both the training code and a Flask API for deployment.

## Features

- Medical certificate image verification using deep learning
- REST API for certificate verification
- Pre-trained model included
- Easy to integrate with other applications

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

## Installation

1. Clone this repository:
```bash
git clone [your-repository-url]
cd [repository-name]
```

2. Create and activate a virtual environment (recommended):
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/MacOS
python -m venv venv
source venv/bin/activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### As a Standalone Service

1. Start the Flask server:
```bash
python app.py
```

The server will start on `http://localhost:5000` by default.

2. Use the API endpoint:
- Endpoint: `POST /verify`
- Content-Type: `multipart/form-data`
- Form parameter: `file` (image file)

Example using curl:
```bash
curl -X POST -F "file=@/path/to/certificate.jpg" http://localhost:5000/verify
```

### Integration in Another Project

1. Install the package in your project:
```bash
pip install -r requirements.txt
```

2. Import and use the verification function:
```python
from verify_certificate import verify_certificate

# Load and verify an image
result = verify_certificate('path/to/image.jpg')
print(result)  # Returns verification result
```

## API Response Format

The API returns JSON responses in the following format:

```json
{
    "status": "success",
    "prediction": true,  // or false
    "confidence": 0.95,  // confidence score
    "message": "Certificate appears to be valid"
}
```

## Model Information

The system uses a pre-trained deep learning model (`medical_certificate_verifier.h5`) based on TensorFlow. The model has been trained on a dataset of medical certificates to distinguish between genuine and fake certificates.

## Error Handling

The API includes proper error handling for:
- Invalid file formats
- Missing files
- Processing errors

## Contributing

Feel free to submit issues and enhancement requests.

## License

[Your License Here]

## Support

For support, please [contact details or link to issues] 