"""
app.py
Servidor Flask que expõe a orquestração multi-agente.
"""

from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from agents import SimpleOrchestrator

load_dotenv()  # carrega variáveis do .env

app = Flask(__name__)
orchestrator = SimpleOrchestrator()


@app.route("/")
def index():
    """Serve a interface gráfica."""
    return render_template("index.html")


@app.route("/orchestrate", methods=["POST"])
def orchestrate():
    """
    Recebe {"text": "..."} via POST e retorna:
    {
        "input_text": "...",
        "groq_sentiment": "...",
        "azure_diagnosis": "..."
    }
    """
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")

    if not text.strip():
        return jsonify({"error": "Campo 'text' vazio ou ausente."}), 400

    result = orchestrator.run(text)

    # Se algo falhou internamente, ainda retornamos 200 com o erro
    # embutido no corpo para o front-end poder exibir de forma amigável.
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)