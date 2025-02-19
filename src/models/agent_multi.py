#!/usr/bin/env python
import os
import sys
import requests
import logging
import json
import time
import itertools
import numpy as np
import mlflow
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from tqdm import tqdm
from dotenv import load_dotenv
import re

# NLTK & Metrics
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# Import autogen components (ensure autogen is installed)
from autogen import UserProxyAgent, AssistantAgent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable in the .env file.")

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK tokenizer if needed
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# Define Model Token Costs (in USD per 1M tokens)
MODEL_COSTS = {
    'gpt-3.5-turbo': {'input': 0.50, 'output': 1.50},
    'gpt-4': {'input': 30.00, 'output': 60.00},
    'gpt-4o': {'input': 2.50, 'output': 10.00},
}

# Embedding model instance (used by ranking agent)
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)


# -----------------------------
# 1. PubMed Search Agent
# -----------------------------
class PubMedSearchAgent(AssistantAgent):
    def __init__(self, name="pubmed_search_agent", model_name="gpt-4o"):
        super().__init__(name, llm_config={"model": model_name, "openai_api_key": OPENAI_API_KEY})
        self.register_function({"search_pubmed": self.search_pubmed})

    def search_pubmed(self, query: str, days_back: int = 7, max_weeks: int = 4, operator: str = "AND") -> dict:
        current_weeks = 1
        articles = []
        while current_weeks <= max_weeks:
            days = current_weeks * 7
            logger.info(f"PubMed search: querying for articles in the last {days} day(s).")
            ids = self.query_pubmed_ids(query, days, operator)
            articles = self.fetch_article_details(ids)
            if articles:
                logger.info(f"Found {len(articles)} articles with {days} day(s) back.")
                break
            else:
                logger.info(f"No articles found with {days} day(s) back; extending search.")
                current_weeks += 1

        if not articles:
            return {"error": "No articles found after extending search."}

        formatted_articles = "\n\n".join([
            f"**Title:** {a['title']}\n**Abstract:** {a['abstract']}\n**Publication Date:** {a.get('date', 'Unknown')}\n**URL:** {a.get('url', '')}"
            for a in articles
        ])
        return {
            "articles": articles,
            "formatted_articles": formatted_articles,
            "days_back": days,
            "num_articles": len(articles)
        }

    def query_pubmed_ids(self, query: str, days_back: int, operator: str) -> list:
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": query,
            "datetype": "pdat",
            "reldate": days_back,
            "retmax": 100,
            "retmode": "json"
        }
        response = requests.get(base_url, params=params)
        logger.info(f"PubMed esearch URL: {response.url}")
        response.raise_for_status()
        time.sleep(3)
        idlist = response.json().get("esearchresult", {}).get("idlist", [])
        logger.info(f"Retrieved {len(idlist)} PubMed IDs for the last {days_back} day(s).")
        return idlist

    def fetch_article_details(self, idlist: list) -> list:
        if not idlist:
            return []
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {
            "db": "pubmed",
            "id": ",".join(idlist),
            "retmode": "xml"
        }
        response = requests.get(base_url, params=params)
        logger.info(f"PubMed efetch URL: {response.url}")
        response.raise_for_status()
        time.sleep(3)
        root = ET.fromstring(response.text)
        articles = []
        for article in root.findall('./PubmedArticle'):
            pmid_el = article.find('.//PMID')
            pmid = pmid_el.text if pmid_el is not None else None
            title_el = article.find('.//ArticleTitle')
            title = title_el.text if title_el is not None else "No Title"
            abstract_el = article.find('.//Abstract/AbstractText')
            abstract = abstract_el.text if abstract_el is not None else "No Abstract"
            pub_date = "Unknown"
            date_el = article.find(".//PubDate/Year")
            if date_el is not None:
                pub_date = date_el.text
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
            articles.append({
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "date": pub_date,
                "url": url
            })
        return articles


# -----------------------------
# 2. Ranking Agent
# -----------------------------
class RankingAgent(AssistantAgent):
    def __init__(self, name="ranking_agent", model_name="gpt-4o"):
        super().__init__(name, llm_config={"model": model_name, "openai_api_key": OPENAI_API_KEY})
        self.register_function({"rank_articles": self.rank_articles})

    def rank_articles(self, articles: list, user_description: str) -> list:
        if not articles:
            return []
        abstracts = []
        for a in articles:
            text = a.get("abstract", "")
            if not isinstance(text, str):
                text = str(text) if text is not None else ""
            if len(text) > 4000:
                text = text[:4000] + "..."
            abstracts.append(text)
        user_embedding = embedding_model.embed_query(user_description)
        article_embeddings = embedding_model.embed_documents(abstracts)
        ranked = []
        for article, emb in zip(articles, article_embeddings):
            sim = self.cosine_similarity(user_embedding, emb)
            article["similarity"] = sim
            ranked.append(article)
        ranked.sort(key=lambda x: x["similarity"], reverse=True)
        return ranked

    def cosine_similarity(self, vec1, vec2) -> float:
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        if not np.any(vec1) or not np.any(vec2):
            return 0.0
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


# -----------------------------
# 3. Summarization Agent (Swarm Members)
# -----------------------------
class SummarizationAgent(AssistantAgent):
    def __init__(self, name, temperature=0.0, summary_word_count=300, prompting_method="Zero-Shot"):
        super().__init__(name,
                         llm_config={"model": "gpt-4o", "temperature": temperature, "openai_api_key": OPENAI_API_KEY})
        self.llm = ChatOpenAI(model_name="gpt-4o", temperature=temperature, openai_api_key=OPENAI_API_KEY)
        self.summary_word_count = summary_word_count
        self.prompting_method = prompting_method
        self.register_function({"summarize_articles": self.summarize_articles})

    def generate_summary_prompt(self, formatted_articles: str, description: str, keywords: list) -> str:
        prompt = (
            f"Summarize the following scientific articles focusing on {description}. "
            f"Highlight key findings, methodologies, and implications related to {', '.join(keywords)}. "
            f"The summary should be approximately {self.summary_word_count} words.\n\n"
            f"{formatted_articles}"
        )
        return prompt

    def summarize_articles(self, articles: list, description: str, keywords: list) -> str:
        max_articles = 10
        if len(articles) > max_articles:
            articles = articles[:max_articles]
        for a in articles:
            abs_text = a.get("abstract", "")
            if not isinstance(abs_text, str):
                abs_text = ""
            if len(abs_text) > 2000:
                a["abstract"] = abs_text[:2000] + "..."
        formatted = "\n\n".join([
            f"**Title:** {a['title']}\n**Abstract:** {a['abstract']}\n**Date:** {a.get('date', 'Unknown')}\n**URL:** {a.get('url', '')}"
            for a in articles
        ])
        prompt = self.generate_summary_prompt(formatted, description, keywords)
        response = self.llm([{"role": "user", "content": prompt}])
        summary = response.content if response and hasattr(response, "content") else "Summarization failed."
        return summary


# -----------------------------
# 4. Decision Agent (Judge)
# -----------------------------
class DecisionAgent(AssistantAgent):
    def __init__(self, name="decision_agent", model_name="gpt-3.5-turbo"):
        super().__init__(name, llm_config={"model": model_name, "temperature": 0.0, "openai_api_key": OPENAI_API_KEY})
        self.register_function({"choose_best_summary": self.choose_best_summary})
        self.judge_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, top_p=1, openai_api_key=OPENAI_API_KEY)
        self.model_name = model_name

    def get_token_count(self, text: str) -> int:
        import tiktoken
        tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        return len(tokenizer.encode(text))

    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        costs = MODEL_COSTS.get(model, {})
        input_cost = costs.get("input", 0.0) * (input_tokens / 1_000_000)
        output_cost = costs.get("output", 0.0) * (output_tokens / 1_000_000)
        return input_cost + output_cost

    def judge_summary(self, summary: str, reference_articles: list, summary_word_count: int) -> dict:
        if not reference_articles:
            logger.warning("No reference articles provided for judging.")
            return {}
        references = "\n\n".join([article['abstract'] for article in reference_articles])
        judge_prompt = (
            f"You are an expert reviewer tasked with evaluating a scientific summary based on the following criteria:\n"
            f"1. **Accuracy**: Does the summary correctly reflect the key points and findings of the articles?\n"
            f"2. **Completeness**: Does the summary include all major findings without omitting essential information?\n"
            f"3. **Clarity**: Is the summary clear, well-structured, and easy to understand?\n"
            f"4. **Conciseness**: Is the summary concise, avoiding unnecessary verbosity while retaining essential details?\n"
            f"5. **Relevance**: Is all included information pertinent to the main topic without introducing unrelated content?\n"
            f"6. **Objectivity**: Is the summary unbiased, presenting information factually without personal opinions or interpretations?\n\n"
            f"**Original Abstracts:**\n{references}\n\n"
            f"**Generated Summary:**\n{summary}\n\n"
            f"For each criterion, provide a score out of 10 and a brief comment. Remember the summary is supposed to be a {summary_word_count}-word high-level summary and will purposefully only discuss key findings while omitting specific details. "
            f"Format the output as a JSON object with each criterion as a key, containing the score and comment. Additionally, provide an overall score out of 60.\n\n"
            f"**Example Output (JSON only, no markdown or code fences):**\n"
            f"{{\n"
            f"  \"Accuracy\": {{\"score\": 9, \"comment\": \"The summary accurately reflects the key findings of the articles.\"}},\n"
            f"  \"Completeness\": {{\"score\": 8, \"comment\": \"Most major aspects are included, but some details are missing.\"}},\n"
            f"  \"Clarity\": {{\"score\": 9, \"comment\": \"The summary is clear and well-structured.\"}},\n"
            f"  \"Conciseness\": {{\"score\": 8, \"comment\": \"The summary is concise and avoids unnecessary verbosity.\"}},\n"
            f"  \"Relevance\": {{\"score\": 9, \"comment\": \"All included information is pertinent to the main topic.\"}},\n"
            f"  \"Objectivity\": {{\"score\": 10, \"comment\": \"The summary is unbiased and presents information factually.\"}},\n"
            f"  \"Overall Score\": 53\n"
            f"}}"
        )
        logger.info("Judging the summary using the Judge Agent.")
        evaluation_response = self.judge_llm([{"role": "user", "content": judge_prompt}])
        evaluation = evaluation_response.content
        logger.info("Judge evaluation completed.")
        logger.debug(f"Raw Judge Evaluation: {evaluation}")
        try:
            evaluation_clean = re.sub(r'^```json\n?|```$', '', evaluation, flags=re.MULTILINE).strip()
            evaluation_dict = json.loads(evaluation_clean)
            logger.debug(f"Parsed Judge Evaluation: {evaluation_dict}")
        except json.JSONDecodeError:
            logger.error("Failed to parse Judge evaluation as JSON.")
            evaluation_dict = {
                "error": "Failed to parse evaluation. Ensure the Judge agent outputs valid JSON.",
                "evaluation": evaluation
            }
        time.sleep(1)
        input_tokens = self.get_token_count(judge_prompt)
        output_tokens = self.get_token_count(evaluation)
        cost = self.calculate_cost(self.model_name, input_tokens, output_tokens)
        logger.info(
            f"Judge Evaluation - Input Tokens: {input_tokens}, Output Tokens: {output_tokens}, Cost: ${cost:.6f}")
        mlflow.log_metric("cost_judge_evaluation", cost)
        return evaluation_dict

    def choose_best_summary(self, summaries: list, reference_articles: list, summary_word_count: int) -> dict:
        best_candidate = summaries[0] if summaries else "No summary generated."
        judge_eval = self.judge_summary(best_candidate, reference_articles, summary_word_count)
        return {"best_summary": best_candidate, "judge_evaluation": judge_eval}


# -----------------------------
# 5. Multi-Agent Orchestrator (Swarm)
# -----------------------------
class MultiAgentOrchestrator(UserProxyAgent):
    def __init__(self, keywords: list, description: str, summary_word_count: int = 300,
                 prompting_methods: list = None, temperatures: list = None, enable_ranking: bool = True):
        super().__init__(name="orchestrator", llm_config={"model": "gpt-4o", "openai_api_key": OPENAI_API_KEY})
        self.keywords = keywords
        self.description = description
        self.summary_word_count = summary_word_count
        self.enable_ranking = enable_ranking
        self.prompting_methods = prompting_methods if prompting_methods else ["Zero-Shot"] * 3
        self.temperatures = temperatures if temperatures else [0.0, 0.5, 1.0]

        self.search_agent = PubMedSearchAgent()
        self.ranking_agent = RankingAgent()
        self.summarization_agents = []
        for i, (temp, prompt_method) in enumerate(zip(self.temperatures, self.prompting_methods)):
            agent_name = f"summarization_agent_{i}"
            self.summarization_agents.append(
                SummarizationAgent(agent_name, temperature=temp,
                                   summary_word_count=self.summary_word_count,
                                   prompting_method=prompt_method)
            )
        self.decision_agent = DecisionAgent()

    def run_pipeline(self, query: str) -> dict:
        search_result = self.search_agent.search_pubmed(query)
        if "error" in search_result:
            return {"error": search_result["error"]}
        articles = search_result["articles"]

        if self.enable_ranking:
            ranked_articles = self.ranking_agent.rank_articles(articles, self.description)
        else:
            ranked_articles = articles

        candidate_summaries = []
        for agent in self.summarization_agents:
            summary = agent.summarize_articles(ranked_articles, self.description, self.keywords)
            candidate_summaries.append(summary)
            time.sleep(1)

        decision = self.decision_agent.choose_best_summary(candidate_summaries, ranked_articles,
                                                           self.summary_word_count)
        best_summary = decision.get("best_summary", "No summary selected.")
        judge_evaluation = decision.get("judge_evaluation", {})

        rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        references = " ".join([a.get("abstract", "") for a in ranked_articles[:10]])
        rouge_scores = rouge.score(references, best_summary)
        summary_tokens = nltk.word_tokenize(best_summary.lower())
        ref_tokens = [nltk.word_tokenize(references.lower())]
        bleu_score = sentence_bleu(ref_tokens, summary_tokens, weights=(0.5, 0.5, 0, 0),
                                   smoothing_function=SmoothingFunction().method1)

        result = {
            "search_result": search_result,
            "ranked_articles": ranked_articles,
            "candidate_summaries": candidate_summaries,
            "best_summary": best_summary,
            "metrics": {
                "ROUGE-1": rouge_scores['rouge1'].fmeasure,
                "ROUGE-2": rouge_scores['rouge2'].fmeasure,
                "ROUGE-L": rouge_scores['rougeL'].fmeasure,
                "BLEU": bleu_score
            },
            "judge_evaluation": judge_evaluation
        }
        return result


# -----------------------------
# 6. Running the Multi-Agent Swarm Experiment
# -----------------------------
def run_experiment_swarm(agent_keywords, agent_description, summary_word_count,
                         prompting_methods, temperatures, enable_ranking, topic_id):
    total_runs = len(prompting_methods) * len(temperatures)
    with tqdm(total=total_runs, desc=f"Swarm Experiments for Topic {topic_id}", unit="run") as pbar:
        for idx, (prompt_method, temp) in enumerate(itertools.product(prompting_methods, temperatures), start=1):
            run_name = f"Topic_{topic_id}_Run_{idx}"
            with mlflow.start_run(run_name=run_name):
                mlflow.log_param("keywords", ", ".join(agent_keywords))
                mlflow.log_param("description", agent_description)
                mlflow.log_param("summary_word_count", summary_word_count)
                mlflow.log_param("prompting_method", prompt_method)
                mlflow.log_param("temperature", temp)
                mlflow.log_param("enable_ranking", enable_ranking)
                mlflow.log_param("framework", "multi_agent_swarm")
                orchestrator = MultiAgentOrchestrator(
                    keywords=agent_keywords,
                    description=agent_description,
                    summary_word_count=summary_word_count,
                    prompting_methods=[prompt_method] * 3,
                    temperatures=[temp, temp, temp],
                    enable_ranking=enable_ranking
                )
                result = orchestrator.run_pipeline(" ".join(agent_keywords))
                if "error" in result:
                    mlflow.log_param("error", result["error"])
                else:
                    mlflow.log_text(result["best_summary"], "best_summary.txt")
                    for i, summary in enumerate(result["candidate_summaries"]):
                        mlflow.log_text(summary, f"candidate_summary_{i}.txt")
                    for metric, value in result["metrics"].items():
                        mlflow.log_metric(metric, value)
                    if result["judge_evaluation"]:
                        for criterion, details in result["judge_evaluation"].items():
                            if criterion != "Overall Score":
                                mlflow.log_metric(f"Judge_{criterion}_Score", details.get("score", 0))
                                mlflow.log_text(details.get("comment", "No comment"), f"Judge_{criterion}_Comment.txt")
                        mlflow.log_metric("Judge_Overall_Score", result["judge_evaluation"].get("Overall Score", 0))
                    mlflow.log_param("num_articles", result["search_result"].get("num_articles", 0))
                pbar.update(1)
                time.sleep(2)


if __name__ == "__main__":
    topics = [
        {
            "keywords": ["quaternary", "ammonium", "compound"],
            "description": "research relating to quaternary ammonium compounds, also known as QACs, and how they combat MRSA",
            "summary_word_count": 300
        },
        {
            "keywords": ["biotechnology", "genomics", "CRISPR"],
            "description": "explorations of CRISPR technology in the field of genomics and its applications in biotechnology",
            "summary_word_count": 300
        },
        {
            "keywords": ["agent", "framework", "artificial"],
            "description": "agent and multiagent based frameworks for artificial intelligence models and systems",
            "summary_word_count": 300
        },
    ]

    prompting_methods = ["Chain of Thought", "Zero-Shot"]
    temperatures = [0.0, 0.5]
    enable_rankings = [True, False]

    mlflow.set_experiment("MultiAgent_Swarm_PubMed_Experiment_v1")
    topic_id = 1
    for topic in topics:
        logger.info(f"Starting swarm experiments for Topic {topic_id}: {', '.join(topic['keywords'])}")
        run_experiment_swarm(
            agent_keywords=topic["keywords"],
            agent_description=topic["description"],
            summary_word_count=topic["summary_word_count"],
            prompting_methods=prompting_methods,
            temperatures=temperatures,
            enable_ranking=enable_rankings,
            topic_id=topic_id
        )
        topic_id += 1

    logger.info("All multi-agent swarm experiments completed. View results in MLflow UI with `mlflow ui`.")
