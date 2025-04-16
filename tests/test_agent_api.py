import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from api.agent_api import app
import pytest
from flask import Flask
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_summarize_success(client):
    try:
        response = client.post("/summarize", json={
            "keywords": ["AI", "protein", "folding"],
            "description": "Recent advancements using artificial intelligence to accurately predict complex protein-folding structures.",
            "model_name": "gpt-3.5-turbo",
            "temperature": 0.5,
            "top_p": 1.0,
            "summary_word_count": 300,
            "prompting_method": "Chain of Thought",
            "enable_ranking": True
        })
        if response.status_code == 429:
            pytest.skip("Skipped due to OpenAI quota issue (429)")
        assert response.status_code == 200
        data = response.get_json()
        assert "summary" in data
    except Exception as e:
        pytest.fail(f"Test failed unexpectedly: {e}")
