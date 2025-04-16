from flask import Flask, request, jsonify
from scirad.models.agent_single import IntelligentAgent
import logging
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env
load_dotenv(find_dotenv(), override=True)

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Add a home route with basic documentation
@app.route("/", methods=["GET"])
def home():
    docs = {
        "message": "Welcome to the Intelligent Agent Summarize API.",
        "description": (
            "This API uses an IntelligentAgent to query PubMed, process and summarize scientific articles. "
            "Use the /summarize endpoint (POST) to generate a summary."
        ),
        "usage": {
            "endpoint": "/summarize",
            "method": "POST",
            "payload_example": {
                "keywords": ["AI", "protein", "folding"],
                "description": "Recent advancements using artificial intelligence to accurately predict complex protein-folding structures.",
                "model_name": "gpt-3.5-turbo",
                "temperature": 0.5,
                "top_p": 1.0,
                "summary_word_count": 300,
                "prompting_method": "Chain of Thought",
                "enable_ranking": True
            }
        },
        "note": "Send the payload as a JSON object in the body of your POST request."
    }
    return jsonify(docs), 200

@app.route("/summarize", methods=["POST"])
def summarize():
    try:
        data = request.json
        keywords = data.get("keywords", [])
        description = data.get("description", "")
        model_name = data.get("model_name", "gpt-3.5-turbo")
        temperature = data.get("temperature", 0.5)
        top_p = data.get("top_p", 1.0)
        summary_word_count = data.get("summary_word_count", 300)
        prompting_method = data.get("prompting_method", "Chain of Thought")
        enable_ranking = data.get("enable_ranking", True)

        agent = IntelligentAgent(
            keywords=keywords,
            description=description,
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
            summary_word_count=summary_word_count,
            prompting_method=prompting_method,
            enable_ranking=enable_ranking
        )

        result = agent.process()

        return jsonify({
            "summary": result["summary"],
            "metrics": result["metrics"],
            "judge_evaluation": result["judge_evaluation"],
            "followup_judgment": result["followup_judgment"],
            "ranking_metric": result["ranking_metric"],
            "similarity_all": result["average_similarity_all"],
            "similarity_top": result["average_similarity_top"]
        }), 200

    except Exception as e:
        logging.exception("Error during processing")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5001)
