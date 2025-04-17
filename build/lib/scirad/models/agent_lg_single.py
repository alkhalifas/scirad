import os
import requests
import logging
import xml.etree.ElementTree as ET
import time
import json
import re
import numexpr  # For secure mathematical evaluations
from dotenv import load_dotenv, find_dotenv

from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
import tiktoken

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define Model Token Costs (in USD per 1M tokens) – if you plan to calculate cost
MODEL_COSTS = {
    'gpt-3.5-turbo': {'input': 0.50, 'output': 1.50},
    'gpt-4': {'input': 30.00, 'output': 60.00},
    'gpt-4o': {'input': 2.50, 'output': 10.00},
}

# Load environment variables from .env file
load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY in your .env file.")


class IntelligentAgent:
    """
    A LangChain Agent that uses multiple tools to search PubMed, perform calculations,
    generate summaries, and even judge summaries—all while using memory and handling errors.
    """

    def __init__(self, keywords, description, model_name="gpt-4o", temperature=0, top_p=1.0,
                 summary_word_count=300, prompting_method="Zero-Shot"):
        # Basic input validations
        if not isinstance(keywords, list) or len(keywords) < 3:
            raise ValueError("Provide at least three keywords as a list.")
        if not isinstance(description, str) or len(description.split()) < 10:
            raise ValueError("Provide a description of at least ten words.")
        if not isinstance(summary_word_count, int) or summary_word_count <= 0:
            raise ValueError("summary_word_count must be a positive integer.")

        self.keywords = keywords
        self.description = description
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.summary_word_count = summary_word_count
        self.prompting_method = prompting_method

        # Initialize the LLM used for all interactions
        self.llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            temperature=self.temperature,
            top_p=self.top_p,
            model_name=self.model_name
        )
        # A separate LLM for judging summaries (if needed)
        self.judge_llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            temperature=0,
            top_p=1,
            model_name="gpt-3.5-turbo"
        )

        # Embedding model for ranking articles
        self.embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

        # Initialize memory for conversation history
        self.memory = ConversationBufferMemory(memory_key="memory", memory_buffer=50)

        # Initialize the tokenizer for token counting
        self.tokenizer = tiktoken.encoding_for_model(self.model_name)

        # Prepare tool list (including error handling in each tool)
        self.tools = [
            Tool(
                name="Calculator",
                func=lambda x: str(numexpr.evaluate(x).item()),
                description="Evaluates mathematical expressions. Provide a valid expression."
            ),
            Tool(
                name="PubMedSearch",
                func=self.pubmed_search_tool,
                description=("Searches PubMed articles. Input format: "
                             "'keywords: cancer, therapy; days_back: 30; operator: AND'")
            ),
            Tool(
                name="SummaryTool",
                func=self.summary_tool,
                description=("Returns a formatted summary report of the most recent PubMed search.")
            )
        ]

        # Initialize the LangChain agent using the Zero-Shot React Description method.
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory,
            handle_parsing_errors=True
        )
        # To store the latest PubMed search results
        self.latest_pubmed_search = {}

    # ------------------------- Utility Methods -------------------------

    def get_token_count(self, text):
        return len(self.tokenizer.encode(text))

    def calculate_cost(self, model, input_tokens, output_tokens):
        costs = MODEL_COSTS.get(model, {})
        input_cost = costs.get('input', 0.0) * (input_tokens / 1_000_000)
        output_cost = costs.get('output', 0.0) * (output_tokens / 1_000_000)
        return input_cost + output_cost

    # ------------------------- PubMed Tools -------------------------

    def query_pubmed_ids(self, keywords, days_back, bool_operator="AND"):
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        joined_keywords = f" {bool_operator} ".join([f'"{kw.strip()}"[Title/Abstract]' for kw in keywords])
        params = {
            "db": "pubmed",
            "term": joined_keywords,
            "datetype": "pdat",
            "reldate": days_back,
            "retmax": 100,
            "retmode": "json"
        }
        response = requests.get(base_url, params=params)
        logger.info(f"eSearch URL: {response.url}")
        response.raise_for_status()
        time.sleep(3)
        esearch_result = response.json().get("esearchresult", {})
        return esearch_result.get("idlist", [])

    def fetch_article_details(self, pubmed_ids):
        if not pubmed_ids:
            return []
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {
            "db": "pubmed",
            "id": ",".join(pubmed_ids),
            "retmode": "xml"
        }
        response = requests.get(base_url, params=params)
        logger.info(f"eFetch URL: {response.url}")
        response.raise_for_status()
        time.sleep(3)
        root = ET.fromstring(response.text)
        articles_info = []
        for pubmed_article in root.findall('./PubmedArticle'):
            pmid_el = pubmed_article.find('.//PMID')
            pmid = pmid_el.text if pmid_el is not None else None
            title_el = pubmed_article.find('.//ArticleTitle')
            title = title_el.text if title_el is not None else "No Title"
            abstract_el = pubmed_article.find('.//Abstract/AbstractText')
            abstract_text = abstract_el.text if abstract_el is not None else "No Abstract"
            pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
            # Simplified publication date retrieval
            date_str = "Unknown"
            pub_date_el = pubmed_article.find(".//MedlineCitation/Article/Journal/JournalIssue/PubDate")
            if pub_date_el is not None:
                year = pub_date_el.find("Year")
                month = pub_date_el.find("Month")
                day = pub_date_el.find("Day")
                date_str = f"{year.text if year is not None else ''}-{month.text if month is not None else ''}-{day.text if day is not None else ''}".strip(
                    "-")
            articles_info.append({
                "pmid": pmid,
                "title": title,
                "abstract": abstract_text,
                "date": date_str,
                "url": pubmed_url
            })
        return articles_info

    def pubmed_search_tool(self, input_text):
        """Tool to search PubMed given a formatted input string."""
        try:
            parts = input_text.split(";")
            keywords_part = next(part for part in parts if "keywords" in part.lower())
            days_back_part = next(part for part in parts if "days_back" in part.lower())
            operator_part = next((part for part in parts if "operator" in part.lower()), "operator: AND")
            keywords = [kw.strip() for kw in keywords_part.split(":")[1].split(",")]
            days_back = int(days_back_part.split(":")[1].strip())
            bool_operator = operator_part.split(":")[1].strip().upper()

            pubmed_ids = self.query_pubmed_ids(keywords, days_back, bool_operator)
            articles = self.fetch_article_details(pubmed_ids)
            if not articles:
                return "No articles found for the given criteria."
            formatted_articles = "\n\n".join([
                f"Title: {article['title']}\nAbstract: {article['abstract']}\nDate: {article['date']}\nURL: {article['url']}"
                for article in articles
            ])
            self.latest_pubmed_search = {
                "keywords": keywords,
                "days_back": days_back,
                "bool_operator": bool_operator,
                "num_articles": len(articles),
                "articles": articles,
                "formatted_articles": formatted_articles
            }
            return formatted_articles

        except Exception as e:
            logger.error(f"PubMedSearch error: {e}")
            return f"An error occurred during PubMed search: {e}"

    def summary_tool(self, input_text):
        """Tool to output a summary report of the latest PubMed search."""
        if not self.latest_pubmed_search:
            return "No PubMed search has been performed yet."
        report = (
            f"Summary Report\n"
            f"--------------\n"
            f"Articles Found: {self.latest_pubmed_search.get('num_articles', 0)}\n"
            f"Days Back Searched: {self.latest_pubmed_search.get('days_back', 'Unknown')}\n\n"
            f"Articles:\n{self.latest_pubmed_search.get('formatted_articles', '')}"
        )
        return report

    # ------------------------- Summarization & Ranking -------------------------

    def generate_summary_prompt(self):
        """Generates a prompt for summarizing articles."""
        return (
            f"Summarize the following articles focusing on {self.description}. "
            f"Create a single-paragraph summary that highlights key findings, methodologies, and implications related to {', '.join(self.keywords)}. "
            f"Keep it around {self.summary_word_count} words."
        )

    def summarize_articles(self):
        base_prompt = self.generate_summary_prompt()
        # In this simple example we directly send the prompt; you might merge with examples based on prompting_method.
        logger.info("Generating summary...")
        response = self.llm([{"role": "user", "content": base_prompt}])
        summary = response.content
        logger.info("Summary generated.")
        # Optionally, log token usage and cost here
        return summary

    def compute_similarity_scores(self, articles):
        if not articles:
            return []
        cleaned = []
        for article in articles:
            text = article.get("abstract", "")
            if not isinstance(text, str):
                text = str(text)
            if len(text) > 4000:
                text = text[:4000] + "..."
            cleaned.append(text)
        user_embedding = self.embedding_model.embed_query(self.description)
        article_embeddings = self.embedding_model.embed_documents(cleaned)
        similarities = []
        for emb in article_embeddings:
            # Compute cosine similarity
            import numpy as np
            vec1 = np.array(user_embedding)
            vec2 = np.array(emb)
            if not np.any(vec1) or not np.any(vec2):
                sim = 0.0
            else:
                sim = float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
            similarities.append(sim)
        return similarities

    # ------------------------- Agent Process -------------------------

    def process(self, query):
        """
        Given a query, decide whether to use a tool (e.g. if the query mentions 'search' or 'calculate')
        or simply let the LLM handle it. Then return the agent's answer.
        """
        # Simple decision logic: if query contains 'search' use PubMedSearch; if 'calculate' use Calculator.
        if "search" in query.lower():
            # Build a search prompt using keywords and a default days_back (e.g. 30 days)
            search_input = f"keywords: {', '.join(self.keywords)}; days_back: 30; operator: AND"
            tool_response = self.agent.run(search_input)
            return tool_response
        elif "calculate" in query.lower():
            # Extract the mathematical expression from the query (this is just an example)
            expression = query.lower().split("calculate")[-1].strip()
            tool_response = self.agent.run(expression)
            return tool_response
        else:
            # Otherwise, pass the query directly to the agent
            return self.agent.run(query)


# ------------------------- Example Usage -------------------------
if __name__ == "__main__":
    # Example inputs
    keywords = ["quaternary", "ammonium", "compounds"]
    description = "Explorations of quaternary ammonium compounds and recent research on that."

    # Initialize the LangChain agent
    agent = IntelligentAgent(
        keywords=keywords,
        description=description,
        model_name="gpt-3.5-turbo",
        temperature=0.2,
        top_p=1.0,
        summary_word_count=300,
        prompting_method="Zero-Shot"
    )

    # Sample interactions
    print("=== PubMed Search ===")
    search_query = "search for latest research"
    result = agent.process(search_query)
    print(result)

