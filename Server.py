<<<<<<< HEAD
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room
import mysql.connector


# Flask 애플리케이션 초기화
app = Flask(__name__)
CORS(app)

socketio = SocketIO(app, cors_allowed_origins = "*")


if __name__ == "__main__":
    socketio.run(app, debug=True, host="0.0.0.0", port=5001)
