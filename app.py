import os
import json
from flask import Flask, render_template, request, jsonify
from google import genai
from google.genai import types

app = Flask(__name__)

# Uses the GEMINI_API_KEY environment variable automatically
client = genai.Client()
MODEL_ID = "gemini-3.5-flash-lite"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    notes = data.get("notes", "").strip()

    if not notes:
        return jsonify({"error": "Please enter some notes or a topic first."}), 400

    try:
        # --- AGENT 1: Concept Extractor & Summarizer ---
        agent1_prompt = (
            "You are an expert academic tutor. Analyze the following text or topic "
            "and produce 4 to 5 concise, high-yield revision bullet points. "
            "Keep explanations clear and exam-focused.\n\n"
            f"INPUT:\n{notes}"
        )
        summary_res = client.models.generate_content(
            model=MODEL_ID,
            contents=agent1_prompt,
        )
        study_notes = summary_res.text

        # --- AGENT 2: Autonomous Quiz Builder ---
        agent2_prompt = (
            "Based ONLY on the study notes below, generate 3 multiple-choice questions "
            "to test a student's understanding.\n\n"
            f"STUDY NOTES:\n{study_notes}"
        )
        quiz_schema = {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "question": {"type": "STRING"},
                    "options": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                    },
                    "answer_index": {
                        "type": "INTEGER",
                        "description": "0-based index of the correct option (0, 1, 2, or 3)",
                    },
                },
                "required": ["question", "options", "answer_index"],
            },
        }

        quiz_res = client.models.generate_content(
            model=MODEL_ID,
            contents=agent2_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=quiz_schema,
            ),
        )
        quiz_data = json.loads(quiz_res.text)

        return jsonify({
            "summary": study_notes,
            "quiz": quiz_data
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)