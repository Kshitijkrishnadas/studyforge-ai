import os
import json
import time
from flask import Flask, render_template, request, jsonify
from google import genai
from google.genai import types
from google.genai.errors import APIError

app = Flask(__name__)

# Uses the GEMINI_API_KEY environment variable automatically
client = genai.Client()

# Primary model first, with automatic fallback models if one is overloaded (503)
MODELS_TO_TRY = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
]


def call_gemini_with_fallback(contents, config=None):
    last_error = None
    for model_id in MODELS_TO_TRY:
        for attempt in range(2):  # up to 2 quick tries per model
            try:
                return client.models.generate_content(
                    model=model_id,
                    contents=contents,
                    config=config,
                )
            except APIError as e:
                last_error = e
                # Retry/fallback on temporary server overload (503) or rate-limit hiccup (429)
                if e.code in (503, 429):
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise
            except Exception as e:
                last_error = e
                raise
    raise last_error


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
           "You are a helpful academic tutor. Analyze the user's input and generate "
            "clear, exam-focused study notes. If the user asks for a detailed or "
            "in-depth explanation, provide a thorough, well-structured breakdown; "
            "otherwise keep it to 4 to 5 concise, high-yield bullet points.\n\n"
            "Formatting rule: Do NOT use LaTeX syntax like $...$, \\ge, or \\rightarrow. "
            "Write clean plain text and standard Unicode symbols instead (such as x², "
            "r², ≥, ≤, →, and 45°).\n\n"
            f"INPUT:\n{notes}"
        )
        summary_res = call_gemini_with_fallback(contents=agent1_prompt)
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

        quiz_res = call_gemini_with_fallback(
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
