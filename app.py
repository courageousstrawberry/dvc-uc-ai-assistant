
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from openai import OpenAI
import os
from dotenv import load_dotenv

# Import necessary functions from ai_agent
from src.ai_agent import (
    llm_parse_user_message,
    load_all_data,
    collect_course_rows,
    filter_rows,
    llm_format_response_multi,
    parse_preferences_seed,
    detect_campuses_from_query
)

load_dotenv()

# --- Flask App Setup ---
# The static_folder is set to 'public' so Flask can serve your index.html, CSS, and JS files.
app = Flask(__name__, static_folder='public', static_url_path='')
CORS(app)

# --- AI Agent and Data Loading ---
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("ERROR: OPENAI_API_KEY is missing. Please set it in your .env file.")
client = OpenAI(api_key=api_key)

# Load all campus data once at startup.
data = load_all_data([
    os.path.join("agreements_25-26", "*.json"),
])
print("✅ Loaded campuses:", sorted(list(data.keys())))
if not data:
    print("⚠️ No campus files loaded. Check the 'agreements_25-26/' directory.")

# --- API and Frontend Routes ---

@app.route('/')
def serve_index():
    """Serves the main index.html file from the 'public' directory."""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/prompt', methods=['POST'])
def handle_prompt():
    """Handles the AI prompt requests from the frontend."""
    if not api_key:
        return jsonify({'error': 'OPENAI_API_KEY is not configured on the server.'}), 500

    req_data = request.get_json()
    user_prompt = req_data.get('prompt')
    if not user_prompt:
        return jsonify({'error': 'No prompt provided.'}), 400

    # 1. Parse the user's message.
    parsed = llm_parse_user_message(client, user_prompt)

    # 2. Determine campuses.
    campus_keys = parsed.get("parameters", {}).get("campuses") or detect_campuses_from_query(user_prompt)
    campus_keys = [ck for ck in campus_keys if ck in data]

    if not campus_keys:
        return jsonify({'response': "Sorry, I couldn't detect a specific campus. I cover UCB, UCD, and UCSD."})

    # 3. Extract filters.
    filters = parsed.get("filters", {})
    completed_courses = set(filters.get("completed_courses", []))
    completed_domains = set(filters.get("domains_completed", []))
    focus_only = filters.get("focus_only")
    required_only = filters.get("required_only", False)
    categories_only = filters.get("categories", [])
    seed_prefs = parse_preferences_seed(user_prompt)

    # 4. Filter course data.
    campus_to_remaining = {}
    for ck in campus_keys:
        all_rows = collect_course_rows(data.get(ck, {}))
        filtered_rows = filter_rows(
            all_rows, seed_prefs, completed_courses, completed_domains,
            focus_only, required_only, categories_only=categories_only
        )
        campus_to_remaining[ck] = filtered_rows

    # 5. Format the final response.
    formatted_response = llm_format_response_multi(
        client, campus_keys, campus_to_remaining, parsed,
        completed_courses, completed_domains, plain=False
    )

    # 6. Send the response back to the frontend.
    return jsonify({'response': formatted_response})

if __name__ == '__main__':
    if not api_key:
        print("FATAL ERROR: The OPENAI_API_KEY environment variable is not set.")
    else:
        print("\nFlask server is running. Open your browser to:")
        print(f"http://127.0.0.1:8080\n")
        app.run(host='0.0.0.0', port=8080, debug=True)