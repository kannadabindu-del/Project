from backend.app import app

if __name__ == "__main__":
    print("🚀 AI Paper Evaluator Starting...")
    print("📍 Open: http://localhost:5000")
    app.run(debug=True, port=5000)