import os
import sys
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.agent_runner import ESGAgent
from src.rag_utils import create_rag_system

def test_rag_fallback():
    rag = create_rag_system(google_api_key=None)
    result = rag.analyze_claim("We achieved 100% carbon neutrality.")
    assert "rag_risk_score" in result
    assert "analysis" in result
    assert result["analysis"].startswith("Keyword-based analysis")

def test_rag_with_gemini(monkeypatch):
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        pytest.skip("No GOOGLE_API_KEY set for Gemini test.")
    rag = create_rag_system(google_api_key=api_key)
    result = rag.analyze_claim("We achieved 100% carbon neutrality.")
    assert "rag_risk_score" in result
    assert "analysis" in result
    # Should not be keyword fallback if LLM is working
    assert not result["analysis"].startswith("Keyword-based analysis")

def test_agent_with_rag(monkeypatch):
    api_key = os.environ.get("GOOGLE_API_KEY")
    from src.agent_runner import ESGAgent
    agent = ESGAgent(
        category_model_path="models/tuned_gradient_boosting_sentence_embeddings.pkl",
        greenwash_gb_model_path="models/tuned_gradient_boosting_sentence_embeddings.pkl",
        greenwash_lr_model_path="models/logistic_regression_greenwashing.pkl",
        google_api_key=api_key
    )
    result = agent.analyze_claim("We achieved 100% carbon neutrality.")
    assert "rag_analysis" in result or "rag_risk_score" in result 