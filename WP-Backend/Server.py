from flask import Flask, request, jsonify
from flask_cors import CORS
import mysql.connector


# Flask 애플리케이션 초기화
app = Flask(__name__)
CORS(app)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)