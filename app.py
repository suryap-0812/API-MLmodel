from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from verify_certificate import verify_certificate
import logging
from functools import wraps
import secrets
from dotenv import load_dotenv
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

try:
    app = Flask(__name__)
    CORS(app)

    # Configure upload settings
    UPLOAD_FOLDER = '/tmp/uploads'  # Change to /tmp for Render
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

    # API Key handling
    API_KEY = os.getenv('API_KEY')
    if not API_KEY:
        # Generate a secure API key if not exists
        API_KEY = secrets.token_urlsafe(32)
        logger.warning("API_KEY not found in environment variables. Generated new key: " + API_KEY)

    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
        logger.info(f"Created upload folder: {UPLOAD_FOLDER}")

    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

    def require_api_key(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            api_key = request.headers.get('X-API-Key')
            if api_key and api_key == API_KEY:
                return f(*args, **kwargs)
            return jsonify({
                'status': 'error',
                'message': 'Invalid or missing API key'
            }), 401
        return decorated_function

    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    @app.route('/', methods=['GET'])
    def index():
        """Root endpoint with API documentation."""
        return jsonify({
            'status': 'success',
            'message': 'Medical Certificate Verification API',
            'endpoints': {
                '/': 'GET - This documentation',
                '/health': 'GET - Health check endpoint',
                '/verify': 'POST - Verify medical certificate (requires X-API-Key header and file upload)'
            }
        })

    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint to verify if the API is running."""
        try:
            return jsonify({
                'status': 'healthy',
                'message': 'Service is running'
            }), 200
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': 'Service health check failed'
            }), 500

    @app.route('/verify', methods=['POST'])
    @require_api_key
    def verify():
        """
        Endpoint to verify a medical certificate.
        Expects a file upload with key 'file'.
        """
        try:
            # Check if file was uploaded
            if 'file' not in request.files:
                return jsonify({
                    'status': 'error',
                    'message': 'No file provided'
                }), 400

            file = request.files['file']

            # Check if a file was selected
            if file.filename == '':
                return jsonify({
                    'status': 'error',
                    'message': 'No file selected'
                }), 400

            # Validate file type
            if not allowed_file(file.filename):
                return jsonify({
                    'status': 'error',
                    'message': f'Invalid file type. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
                }), 400

            # Save file temporarily
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            logger.info(f"File saved temporarily at: {filepath}")

            try:
                # Verify the certificate
                result = verify_certificate(filepath)
                logger.info(f"Certificate verification result: {result}")
                
                # Clean up the temporary file
                os.remove(filepath)
                logger.info(f"Temporary file removed: {filepath}")

                return jsonify({
                    'status': 'success',
                    'prediction': bool(result['prediction']),
                    'confidence': float(result['confidence']),
                    'message': 'Certificate appears to be valid' if result['prediction'] else 'Certificate appears to be invalid'
                }), 200

            except Exception as e:
                # Clean up the temporary file in case of processing error
                if os.path.exists(filepath):
                    os.remove(filepath)
                logger.error(f"Error processing file: {str(e)}")
                raise e

        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': 'Error processing the certificate'
            }), 500

    @app.errorhandler(413)
    def request_entity_too_large(error):
        """Handle file size too large error."""
        return jsonify({
            'status': 'error',
            'message': 'File too large. Maximum size is 16MB'
        }), 413

    @app.errorhandler(500)
    def internal_server_error(error):
        """Handle internal server errors."""
        return jsonify({
            'status': 'error',
            'message': 'Internal server error'
        }), 500

    @app.errorhandler(404)
    def not_found_error(error):
        """Handle 404 errors."""
        return jsonify({
            'status': 'error',
            'message': 'Endpoint not found'
        }), 404

except Exception as e:
    logger.error(f"Error during app initialization: {str(e)}")
    raise e

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False) 