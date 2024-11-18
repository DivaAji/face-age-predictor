from flask import Flask
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    CORS(app)  # Tambahkan jika aplikasi membutuhkan akses dari domain berbeda
    
    # Daftarkan blueprint
    from .routes import main
    app.register_blueprint(main)
    
    return app
