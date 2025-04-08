# openai_intelligent_agent.py

import os
import sys
import requests
import logging
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI  # Updated import
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.embeddings import OpenAIEmbeddings
import numexpr  # For secure mathematical evaluations
import json
import re
from datetime import datetime, timedelta
import mlflow
import time
import itertools
from tqdm import tqdm
from dotenv import load_dotenv, find_dotenv

# Metrics Libraries
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Token Counting Library
import tiktoken

# Initialize NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

load_dotenv(find_dotenv(), override=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define Model Token Costs (in USD per 1M tokens)
MODEL_COSTS = {
    'gpt-3.5-turbo': {'input': 0.50, 'output': 1.50},
    'gpt-4': {'input': 30.00, 'output': 60.00},
    'gpt-4o': {'input': 2.50, 'output': 10.00},
    # Add other models and their costs here if needed
}

class IntelligentAgent:
    """
    A class to encapsulate the functionalities of the Intelligent Agent,
    including PubMed search, ranking, summary generation, metrics calculation, Judge assessment, and user interaction.
    """

    def __init__(self, keywords, description, model_name="gpt-4o", temperature=0, top_p=1.0,
                 max_search_weeks=4, search_extension_step=1, summary_word_count=100,
                 prompting_method="Zero-Shot", enable_ranking=True):
        """
        Initializes the Intelligent Agent by loading environment variables,
        setting up the language models, tools, memory, and agent.

        Parameters:
            keywords (list): List of keywords (minimum 3).
            description (str): Description of the research focus (minimum 10 words).
            model_name (str): Name of the OpenAI model to use.
            temperature (float): Sampling temperature.
            top_p (float): Nucleus sampling parameter.
            max_search_weeks (int): Maximum number of weeks to extend the search.
            search_extension_step (int): Number of weeks to extend each time.
            summary_word_count (int): Desired number of words in the summary.
            prompting_method (str): The prompting method to use for summarization.
            enable_ranking (bool): Flag to enable or disable the ranking of articles based on similarity.
        """
        # Validate user inputs
        if not isinstance(keywords, list) or len(keywords) < 3:
            raise ValueError("Please provide at least three keywords as a list.")
        if not isinstance(description, str) or len(description.split()) < 10:
            raise ValueError("Please provide a description of at least ten words.")
        if not isinstance(summary_word_count, int) or summary_word_count <= 0:
            raise ValueError("Please provide a positive integer for summary_word_count.")

        self.keywords = keywords
        self.description = description
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_search_weeks = max_search_weeks
        self.search_extension_step = search_extension_step
        self.summary_word_count = summary_word_count
        self.prompting_method = prompting_method  # Existing parameter
        self.enable_ranking = enable_ranking  # New parameter

        # Log the ranking flag for debugging
        logger.info(f"Ranking Enabled: {self.enable_ranking}")

        # Validate the prompting_method
        supported_methods = [
            "Chain of Thought",
            "Tree of Thought",
            "Few-Shot",
            "Zero-Shot",
            "Instruction-Based",
            "Role-Based",
            # Add other supported methods here
        ]
        if self.prompting_method not in supported_methods:
            raise ValueError(f"Unsupported prompting method: {self.prompting_method}. "
                             f"Supported methods are: {supported_methods}")

        # Load environment variables from .env file
        load_dotenv()

        # Fetch API keys from environment variables
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        if not self.openai_api_key:
            raise ValueError("Please set the OPENAI_API_KEY environment variable in the .env file.")

        # Initialize OpenAI LLM for Summarization
        self.llm = ChatOpenAI(
            openai_api_key=self.openai_api_key,
            temperature=self.temperature,
            top_p=self.top_p,
            model_name=self.model_name  # Dynamic model selection
        )

        # Initialize OpenAI LLM for Judge Agent
        self.judge_llm = ChatOpenAI(
            openai_api_key=self.openai_api_key,
            temperature=0,  # Can vary separately if needed
            top_p=1,
            model_name="gpt-3.5-turbo"
        )

        # Initialize Embedding Model for Ranking
        self.embedding_model = OpenAIEmbeddings(openai_api_key=self.openai_api_key)

        # Initialize Memory
        self.memory = ConversationBufferMemory(memory_key="memory", memory_buffer=50)  # Adjust 'memory_buffer' as needed

        # Initialize Tools
        self.tools = [
            Tool(
                name="Calculator",
                func=lambda x: str(numexpr.evaluate(x).item()),
                description="Useful for performing mathematical calculations. Input should be a valid mathematical expression."
            ),
            Tool(
                name="PubMedSearch",
                func=self.pubmed_search_tool,
                description=(
                    "Useful for searching PubMed articles. "
                    "Input should specify keywords, days back, and a boolean operator in the following format: "
                    "'keywords: cancer, therapy; days_back: 30; operator: AND'"
                )
            ),
            Tool(
                name="SummaryTool",
                func=self.summary_tool,
                description=(
                    "Provides a summary report of the latest PubMed search. "
                    "The report includes the number of articles summarized, the number of days back searched, and the final summary of articles."
                )
            )
        ]

        # Initialize Agent
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Changed AgentType
            verbose=True,
            memory=self.memory
        )

        # Initialize PubMed search storage
        self.latest_pubmed_search = {}

        # Initialize prompt storage
        self.agent_prompt = ""
        self.judge_prompt = ""

        # Initialize tokenizer
        self.tokenizer = tiktoken.encoding_for_model(self.model_name)

    def get_prompt_template(self, prompting_method):
        """
        Retrieves the prompt template based on the specified prompting method.

        Parameters:
            prompting_method (str): The name of the prompting method to use.

        Returns:
            str: The formatted prompt template.
        """
        if prompting_method == "Chain of Thought":
            return self.chain_of_thought_prompt()
        elif prompting_method == "Tree of Thought":
            return self.tree_of_thought_prompt()
        elif prompting_method == "Few-Shot":
            return self.few_shot_prompt()
        elif prompting_method == "Zero-Shot":
            return self.zero_shot_prompt()
        elif prompting_method == "Instruction-Based":
            return self.instruction_based_prompt()
        elif prompting_method == "Role-Based":
            return self.role_based_prompt()
        else:
            raise ValueError(f"Unsupported prompting method: {prompting_method}")

    def chain_of_thought_prompt(self):
        return (
            "You are an expert scientific researcher. Please follow a step-by-step reasoning process to summarize the following articles.\n\n"
            "{formatted_articles}\n\n"
            "Step 1: Identify the key findings from each article.\n"
            "Step 2: Analyze the methodologies used.\n"
            "Step 3: Discuss the implications related to the keywords.\n"
            "Step 4: Compose a coherent summary based on the above analysis. Include specific details from the papers you see.\n\n"
            "The summary should be approximately {summary_word_count} words."
        )

    def tree_of_thought_prompt(self):
        return (
            "As a scientific analyst, explore multiple reasoning paths to summarize the following articles.\n\n"
            "{formatted_articles}\n\n"
            "Consider different aspects such as key findings, methodologies, and implications. "
            "Develop multiple drafts based on these aspects and select the most comprehensive and coherent summary.\n\n"
            "The final summary should be approximately {summary_word_count} words. Include specific details from the papers you see."
        )

    def few_shot_prompt(self):
        return (
            "You are an experienced researcher tasked with summarizing scientific articles. Below are examples of summaries:\n\n"
            "Example 1:\n"
            "Article: {article_1_title}\nAbstract: {article_1_abstract}\nSummary: {article_1_summary}\n\n"
            "Example 2:\n"
            "Article: {article_2_title}\nAbstract: {article_2_abstract}\nSummary: {article_2_summary}\n\n"
            "Now, summarize the following articles:\n\n"
            "{formatted_articles}\n\n"
            "The summary should be approximately {summary_word_count} words."
        )

    def zero_shot_prompt(self):
        return (
            "Summarize the following scientific articles focusing on {description}. "
            "Highlight key findings, methodologies, and implications related to {keywords}. "
            "The summary should be approximately {summary_word_count} words."
        )

    def instruction_based_prompt(self):
        return (
            "You are a scientific summary generator. Please create a concise summary of the provided articles with the following instructions:\n"
            "- Focus on {description}.\n"
            "- Highlight key findings, methodologies, and implications related to {keywords}.\n"
            "- The summary should be around {summary_word_count} words.\n"
            "- Ensure clarity and conciseness."
        )

    def role_based_prompt(self):
        return (
            "As a seasoned scientific reviewer, please summarize the following articles. "
            "Ensure that your summary covers key findings, methodologies, and implications related to {keywords}. "
            "Keep the summary approximately {summary_word_count} words long."
        )

    def get_token_count(self, text):
        """
        Counts the number of tokens in a given text using the model's tokenizer.

        Parameters:
            text (str): The text to tokenize.

        Returns:
            int: Number of tokens.
        """
        return len(self.tokenizer.encode(text))

    def calculate_cost(self, model, input_tokens, output_tokens):
        """
        Calculates the cost based on the number of input and output tokens.

        Parameters:
            model (str): The model name.
            input_tokens (int): Number of input tokens.
            output_tokens (int): Number of output tokens.

        Returns:
            float: Total cost in USD.
        """
        costs = MODEL_COSTS.get(model, {})
        input_cost = costs.get('input', 0.0) * (input_tokens / 1_000_000)
        output_cost = costs.get('output', 0.0) * (output_tokens / 1_000_000)
        total_cost = input_cost + output_cost
        return total_cost

    def query_pubmed_ids(self, keywords, days_back, bool_operator="AND"):
        """
        Queries the NCBI E-utilities API (esearch) to retrieve PubMed IDs based on keywords and date range.

        Parameters:
            keywords (list): List of keywords to search for.
            days_back (int): Number of days back from the current date to search.
            bool_operator (str): Boolean operator to combine keywords ("AND" or "OR").

        Returns:
            list: List of retrieved PubMed IDs.
        """
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
        logger.info(f">> eSearch URL: {response.url}")
        logger.info(f">> HTTP status code: {response.status_code}")

        response.raise_for_status()

        # **Add Sleep Here**
        time.sleep(3)  # Sleep for 3 seconds

        esearch_result = response.json().get("esearchresult", {})
        idlist = esearch_result.get("idlist", [])

        logger.info(f"Retrieved {len(idlist)} PubMed IDs for the last {days_back} day(s).")
        return idlist

    def fetch_article_details(self, pubmed_ids):
        """
        Retrieves detailed information for each PubMed ID using the eFetch API.

        Parameters:
            pubmed_ids (list): List of PubMed IDs to fetch details for.

        Returns:
            list: List of dictionaries containing article details.
        """
        if not pubmed_ids:
            return []

        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {
            "db": "pubmed",
            "id": ",".join(pubmed_ids),
            "retmode": "xml"
        }

        response = requests.get(base_url, params=params)
        logger.info(f">> eFetch URL: {response.url}")
        logger.info(f">> HTTP status code: {response.status_code}")

        response.raise_for_status()

        # **Add Sleep Here**
        time.sleep(3)  # Sleep for 3 seconds

        root = ET.fromstring(response.text)
        articles_info = []

        for pubmed_article in root.findall('./PubmedArticle'):
            # PMID
            pmid_el = pubmed_article.find('.//PMID')
            pmid = pmid_el.text if pmid_el is not None else None

            # Title
            title_el = pubmed_article.find('.//ArticleTitle')
            title = title_el.text if title_el is not None else "No Title"

            # Abstract
            abstract_el = pubmed_article.find('.//Abstract/AbstractText')
            abstract_text = abstract_el.text if abstract_el is not None else "No Abstract"

            # PubMed URL
            pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""

            # Publication date
            date_str = "Unknown"
            pub_date_el = pubmed_article.find(".//MedlineCitation/Article/Journal/JournalIssue/PubDate")
            if pub_date_el is not None:
                year_el = pub_date_el.find("Year")
                month_el = pub_date_el.find("Month")
                day_el = pub_date_el.find("Day")

                year_val = year_el.text if year_el is not None else ""
                month_val = month_el.text if month_el is not None else ""
                day_val = day_el.text if day_el is not None else ""

                season_el = pub_date_el.find("Season")
                if season_el is not None:
                    month_val = season_el.text

                if year_val or month_val or day_val:
                    date_str = f"{year_val}-{month_val}-{day_val}".strip("-")

            articles_info.append({
                "pmid": pmid,
                "title": title,
                "abstract": abstract_text,
                "date": date_str,
                "url": pubmed_url
            })

        return articles_info

    def pubmed_search_tool(self, input_text):
        """
        Searches PubMed based on user input and updates the latest search results.

        Parameters:
            input_text (str): User input specifying keywords, days_back, and operator.

        Returns:
            str: Formatted summary of retrieved articles or an error message.
        """
        try:
            # Parsing the input_text
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

            # Format the articles for better readability
            formatted_articles = "\n\n".join([
                f"**Title:** {article['title']}\n"
                f"**Abstract:** {article['abstract']}\n"
                f"**Publication Date:** {article['date']}\n"
                f"**URL:** {article['url']}"
                for article in articles
            ])

            # Update the latest_pubmed_search dictionary
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
            logger.error(f"Error in PubMedSearch Tool: {e}")
            return f"An error occurred while searching PubMed: {e}"

    def generate_search_prompt(self, current_weeks_back):
        """
        Generates a PubMed search prompt based on keywords and description.

        Parameters:
            current_weeks_back (int): Number of weeks back to search.

        Returns:
            str: Formatted search prompt.
        """
        prompt = (
            f"keywords: {', '.join(self.keywords)}; "
            f"days_back: {current_weeks_back * 7}; "
            f"operator: AND"
        )
        return prompt

    def generate_summary_prompt(self):
        """
        Generates a summary prompt incorporating keywords, description, and desired word count.

        Returns:
            str: Formatted summary prompt.
        """
        prompt_template = (
            f"Summarize the following articles focusing on {self.description}. "
            f"The summary is meant to be a SINGLE paragraph that summarizes the findings of the article's findings"
            f"Start the summary with something like 'recent research has shown...'"
            f"Make sure the summary is only one paragraph. Do not refer to the articles as 'Article 1', just discuss the discovery."
            f"At the end of the summary, include a list of the articles along with their URLs. "
            f"Ensure that the summary highlights the key findings, methodologies, and implications related to {', '.join(self.keywords)}. "
            f"The summary should be approximately {self.summary_word_count} words, suitable for a weekly update, emphasizing key findings while including specific details. Include specific details from the papers you see."
        )
        return prompt_template

    def extend_search_if_no_results(self):
        current_weeks_back = 1  # Start with 1 week
        while current_weeks_back <= self.max_search_weeks:
            logger.info(f"Searching PubMed for the last {current_weeks_back} week(s).")
            search_prompt = self.generate_search_prompt(current_weeks_back)
            articles = self.pubmed_search_tool(search_prompt)
            # If we get real articles, break out
            if articles != "No articles found for the given criteria.":
                logger.info(f"Articles found with {current_weeks_back} week(s) back.")
                # At this point, self.latest_pubmed_search["days_back"]
                # should be current_weeks_back * 7 inside pubmed_search_tool()
                return True
            else:
                logger.info(f"No articles found with {current_weeks_back} week(s) back. Extending search.")
                current_weeks_back += self.search_extension_step
        logger.warning(f"No articles found even after {self.max_search_weeks} weeks.")
        return False


    def compute_similarity_scores(self, articles):
        """
        Computes similarity scores between the user's description and each article's abstract.
        """
        if not articles:
            return []

        # Convert and truncate each abstract
        cleaned_abstracts = []
        for article in articles:
            text = article['abstract']
            # 1) Cast to string if None or any other type:
            if not isinstance(text, str):
                text = "" if text is None else str(text)
            # 2) Truncate if it's extremely long (e.g. over 4000 chars)
            max_chars = 4000
            if len(text) > max_chars:
                text = text[:max_chars] + "..."
            cleaned_abstracts.append(text)

        # Generate embeddings
        logger.info("Generating embeddings for user description and article abstracts.")
        user_embedding = self.embedding_model.embed_query(self.description)
        abstract_embeddings = self.embedding_model.embed_documents(cleaned_abstracts)

        # Compute cosine similarities
        logger.info("Computing cosine similarity between user description and article abstracts.")
        similarities = []
        for abstract_embedding in abstract_embeddings:
            similarity = self.cosine_similarity(user_embedding, abstract_embedding)
            similarities.append(similarity)

        return similarities


    def rank_articles(self, articles):
        """
        Ranks articles based on their relevance to the user's description using semantic similarity.
        Returns a list of articles sorted by similarity in descending order.
        """
        if not articles:
            logger.warning("No articles to rank.")
            return []

        # --- 1) Cast/truncate each abstract ---
        cleaned_abstracts = []
        max_chars = 4000  # Adjust if needed
        for article in articles:
            text = article["abstract"]
            # Ensure it's a string (handle None or other data types)
            if not isinstance(text, str):
                text = "" if text is None else str(text)
            # Optionally truncate if it's extremely long
            if len(text) > max_chars:
                text = text[:max_chars] + "..."
            cleaned_abstracts.append(text)

        # --- 2) Generate embeddings for user description and each abstract ---
        logger.info("Generating embeddings for user description and article abstracts.")
        user_embedding = self.embedding_model.embed_query(self.description)
        abstract_embeddings = self.embedding_model.embed_documents(cleaned_abstracts)

        # --- 3) Compute similarity scores ---
        logger.info("Computing cosine similarity between user description and article abstracts.")
        similarities = []
        for idx, abstract_embedding in enumerate(abstract_embeddings):
            similarity = self.cosine_similarity(user_embedding, abstract_embedding)
            similarities.append(similarity)

        # --- 4) Attach similarity scores to the articles ---
        for idx, article in enumerate(articles):
            article["similarity"] = similarities[idx]

        # --- 5) Sort articles by similarity (descending) ---
        ranked_articles = sorted(articles, key=lambda x: x["similarity"], reverse=True)
        if ranked_articles:
            logger.info(f"Articles ranked based on similarity. "
                        f"Top similarity score: {ranked_articles[0]['similarity']:.4f}")
        else:
            logger.info("No articles returned after ranking.")

        return ranked_articles


    def cosine_similarity(self, vec1, vec2):
        """
        Computes the cosine similarity between two vectors.

        Parameters:
            vec1 (list or np.array): First vector.
            vec2 (list or np.array): Second vector.

        Returns:
            float: Cosine similarity score.
        """
        import numpy as np
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        if not np.any(vec1) or not np.any(vec2):
            return 0.0
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

    def compute_ranking_metric(self, ranked_articles, top_n=10):
        """
        Computes the average cosine similarity of the top N ranked articles.

        Parameters:
            ranked_articles (list): List of ranked articles sorted by similarity.
            top_n (int): Number of top articles to consider.

        Returns:
            float: Average cosine similarity score of the top N articles.
        """
        if not ranked_articles:
            logger.warning("No articles to compute ranking metric.")
            return 0.0

        top_articles = ranked_articles[:top_n]
        total_similarity = sum(article.get('similarity', 0.0) for article in top_articles)
        average_similarity = total_similarity / len(top_articles)

        logger.info(f"Average similarity of top {top_n} articles: {average_similarity:.4f}")
        return average_similarity

    def summary_tool(self, input_text):
        """
        Generates a summary report of the latest PubMed search.

        Parameters:
            input_text (str): Formatted articles summary.

        Returns:
            str: The summary report.
        """
        if not self.latest_pubmed_search:
            return "No PubMed search has been performed yet."

        num_articles = self.latest_pubmed_search.get("num_articles", 0)
        days_back = self.latest_pubmed_search.get("days_back", "Unknown")
        formatted_articles = self.latest_pubmed_search.get("formatted_articles", "No articles to summarize.")

        summary_report = (
            f"**Summary Report**\n"
            f"-------------------\n"
            f"**Number of Articles Summarized:** {num_articles}\n"
            f"**Days Back Searched:** {days_back}\n\n"
            f"**Articles Summary:**\n{formatted_articles}"
        )

        return summary_report

    def summarize_articles(self):
        """
        Generates a summary of the retrieved articles using the LLM.
        Returns:
            str: The generated summary.
        """
        # Retrieve the base summary prompt
        base_prompt = self.generate_summary_prompt()

        # Retrieve the structured prompt method
        custom_prompt_method = self.get_prompt_template(self.prompting_method)

        # Ensure they are combined properly
        final_prompt = f"{base_prompt}\n\n{custom_prompt_method}".format(
            formatted_articles=self.latest_pubmed_search.get("formatted_articles", ""),
            description=self.description,
            keywords=", ".join(self.keywords),
            summary_word_count=self.summary_word_count
        )

        logger.info("Generating summary using the LLM with the selected prompting method.")

        # Send the final combined prompt to the LLM
        summary_response = self.llm([
            {"role": "user", "content": final_prompt}
        ])

        summary = summary_response.content
        logger.info("Summary generation completed.")

        # Sleep for a second to avoid rapid API calls
        time.sleep(1)

        # Calculate token usage and cost
        input_tokens = self.get_token_count(final_prompt)
        output_tokens = self.get_token_count(summary)
        cost = self.calculate_cost(self.model_name, input_tokens, output_tokens)

        logger.info(f"Summary Generation - Input Tokens: {input_tokens}, Output Tokens: {output_tokens}, Cost: ${cost:.6f}")

        # Store the cost for later retrieval
        self.last_cost = cost

        # Log cost as a metric in MLflow
        mlflow.log_metric("cost_summary_generation", cost)

        return summary


    def compute_metrics(self, summary):
        if not self.latest_pubmed_search.get("articles"):
            logger.warning("No articles to compute metrics against.")
            return {}

        # Convert each abstract to a safe string
        references = []
        for article in self.latest_pubmed_search['articles']:
            abs_text = article['abstract']
            if not isinstance(abs_text, str):
                abs_text = "" if abs_text is None else str(abs_text)
            references.append(abs_text)

        # Now references is guaranteed to be strings, so join and tokenize safely
        joined_refs = " ".join(references).lower()
        summary_lower = summary.lower()

        # Tokenize summary for BLEU
        summary_tokens = nltk.word_tokenize(summary_lower)

        # Initialize ROUGE scorer
        rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = rouge.score(joined_refs, summary_lower)

        # Compute BLEU score
        references_tokens = [nltk.word_tokenize(ref.lower()) for ref in references]
        bleu_score = sentence_bleu(
            references_tokens,
            summary_tokens,
            weights=(0.5, 0.5, 0, 0),
            smoothing_function=SmoothingFunction().method1
        )

        metrics = {
            "ROUGE-1": rouge_scores['rouge1'].fmeasure,
            "ROUGE-2": rouge_scores['rouge2'].fmeasure,
            "ROUGE-L": rouge_scores['rougeL'].fmeasure,
            "BLEU": bleu_score
        }

        logger.info(f"Computed Metrics: {metrics}")
        return metrics


    def judge_summary(self, summary):
        """
        Uses a separate LLM instance to assess the quality of the summary relative to the original abstracts.

        Parameters:
            summary (str): The generated summary.

        Returns:
            dict: Dictionary containing scores for each criterion and overall feedback.
        """
        if not self.latest_pubmed_search.get("articles"):
            logger.warning("No articles to judge against.")
            return {}

        # Concatenate all abstracts as a single reference
        references = "\n\n".join([article['abstract'] for article in self.latest_pubmed_search['articles']])

        # Define the Judge prompt
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
            f"For each criterion, provide a score out of 10 and a brief comment. Remember the summary is supposed to be a {self.summary_word_count}-word high-level summary and will purposefully only discuss key findings while omitting specific details."
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

        self.judge_prompt = judge_prompt  # Store the Judge prompt

        logger.info("Judging the summary using the Judge Agent.")
        # Modify the prompt to fit the expected message format
        evaluation_response = self.judge_llm([
            {"role": "user", "content": judge_prompt}
        ])

        evaluation = evaluation_response.content
        logger.info("Judge evaluation completed.")

        # Log the raw evaluation for debugging
        logger.debug(f"Raw Judge Evaluation: {evaluation}")

        try:
            # Remove code fences if present
            evaluation_clean = re.sub(r'^```json\n?|```$', '', evaluation, flags=re.MULTILINE).strip()
            evaluation_dict = json.loads(evaluation_clean)
            logger.debug(f"Parsed Judge Evaluation: {evaluation_dict}")
        except json.JSONDecodeError:
            # If JSON parsing fails, return raw evaluation
            logger.error("Failed to parse Judge evaluation as JSON.")
            evaluation_dict = {
                "error": "Failed to parse evaluation. Ensure the Judge agent outputs valid JSON.",
                "evaluation": evaluation
            }

        # **Add Sleep Here**
        time.sleep(1)  # Sleep for 1 second

        # Calculate token usage and cost for judge evaluation
        input_tokens = self.get_token_count(judge_prompt)
        output_tokens = self.get_token_count(evaluation)
        cost = self.calculate_cost(self.model_name, input_tokens, output_tokens)

        logger.info(f"Judge Evaluation - Input Tokens: {input_tokens}, Output Tokens: {output_tokens}, Cost: ${cost:.6f}")

        # Log cost as a metric in MLflow
        mlflow.log_metric("cost_judge_evaluation", cost)

        return evaluation_dict

    def followup_judge(self, summary, initial_judgment):
        """
        Asks the judge to reconsider its initial evaluation by providing it with the summary, the original abstracts,
        and the first round of scores. Returns the follow-up evaluation.
        """
        # Concatenate abstracts as a single reference string
        references = "\n\n".join([article['abstract'] for article in self.latest_pubmed_search['articles']])

        # Construct a follow-up prompt that includes the summary, the original abstracts, and the initial judge evaluation.
        followup_prompt = (
            f"You previously evaluated the following scientific summary:\n\n"
            f"Summary:\n{summary}\n\n"
            f"Original Abstracts:\n{references}\n\n"
            f"And your initial evaluation was:\n{json.dumps(initial_judgment, indent=2)}\n\n"
            f"Please review the summary in the context of the abstracts and your previous evaluation. "
            f"Re-evaluate the summary. Are you sure about your scores and assessments? If not, adjust your scores accordingly and provide an updated JSON output "
            f"with the same criteria as before."
        )

        # Save the follow-up prompt for logging if desired
        self.judge_prompt += "\n\nFollow-up Prompt:\n" + followup_prompt

        logger.info("Sending follow-up prompt to the Judge Agent.")
        followup_response = self.judge_llm([
            {"role": "user", "content": followup_prompt}
        ])

        followup_evaluation = followup_response.content
        logger.info("Follow-up Judge evaluation completed.")

        try:
            followup_clean = re.sub(r'^```json\n?|```$', '', followup_evaluation, flags=re.MULTILINE).strip()
            followup_dict = json.loads(followup_clean)
        except json.JSONDecodeError:
            logger.error("Failed to parse follow-up Judge evaluation as JSON.")
            followup_dict = {
                "error": "Failed to parse follow-up evaluation. Ensure the Judge agent outputs valid JSON.",
                "evaluation": followup_evaluation
            }

        # Optionally log token usage and cost
        input_tokens = self.get_token_count(followup_prompt)
        output_tokens = self.get_token_count(followup_evaluation)
        cost = self.calculate_cost(self.model_name, input_tokens, output_tokens)
        logger.info(
            f"Follow-up Judge Evaluation - Input Tokens: {input_tokens}, Output Tokens: {output_tokens}, Cost: ${cost:.6f}")
        mlflow.log_metric("cost_followup_judge_evaluation", cost)

        return followup_dict


    def recommend_keywords(self, user_description, n=5):
        """
        Recommends n keywords based on the user's description.

        Parameters:
            user_description (str): The description provided by the user.
            n (int): The number of keywords to recommend.

        Returns:
            list: A list of recommended keywords.
        """
        # Create a prompt asking for keywords
        prompt = (
            f"Based on the following description, recommend {n} relevant keywords for research:\n\n"
            f"The keywords should be single words, and you should only recommend 3 words"
            f"Description: {user_description}\n\n"
            "Please provide the keywords as a comma-separated list."
        )

        # Call the LLM using the existing llm instance
        response = self.llm([
            {"role": "user", "content": prompt}
        ])

        # Extract the content and process it
        keywords_text = response.content.strip()

        # Attempt to split the response into keywords
        # Assuming the response is a comma-separated list of keywords
        keywords = [kw.strip() for kw in keywords_text.split(",") if kw.strip()]

        # If the LLM returns fewer keywords than requested, log a warning
        if len(keywords) < n:
            logger.warning("LLM returned fewer keywords than requested.")

        return keywords

    def process(self):
        """
        Orchestrates the entire process: searching, ranking (if enabled), summarizing, computing metrics, and judging.
        Includes logic to limit articles to the top N and truncate abstracts to prevent token overflow.
        """
        # Step 1: Search for articles (with automatic search extension)
        articles_found = self.extend_search_if_no_results()
        if not articles_found:
            agent_prompt = self.generate_summary_prompt()
            return {
                "summary": "No relevant articles found after extending the search period.",
                "metrics": {},
                "judge_evaluation": {},
                "ranking_metric": 0.0,
                "average_similarity_all": 0.0,
                "average_similarity_top": 0.0,
                "agent_prompt": agent_prompt,  # Include Agent prompt used in the no-articles case
                "judge_prompt": ""  # No Judge prompt as there's no summary
            }

        # Step 1.5: Compute similarity for all articles
        all_similarities = self.compute_similarity_scores(self.latest_pubmed_search['articles'])
        average_similarity_all = sum(all_similarities) / len(all_similarities) if all_similarities else 0.0
        logger.info(f"Average similarity of all retrieved articles: {average_similarity_all:.4f}")

        # Step 2: Rank articles if enabled
        if self.enable_ranking:
            ranked_articles = self.rank_articles(self.latest_pubmed_search['articles'])
            ranking_metric = self.compute_ranking_metric(ranked_articles, top_n=10)

            # Compute average similarity for top N
            top_n = 10
            top_articles = ranked_articles[:top_n]
            top_similarities = self.compute_similarity_scores(top_articles)
            average_similarity_top = sum(top_similarities) / len(top_similarities) if top_similarities else 0.0
            logger.info(f"Average similarity of top {top_n} ranked articles after recalculation: {average_similarity_top:.4f}")
        else:
            # No ranking: keep them as-is but set default metrics
            ranked_articles = self.latest_pubmed_search['articles']
            ranking_metric = 0.0
            average_similarity_top = average_similarity_all

        # Step 2.5: Optionally limit the final summarization to the top N articles (to avoid huge contexts)
        max_articles_for_summary = 10
        if len(ranked_articles) > max_articles_for_summary:
            logger.info(f"Limiting summarization to top {max_articles_for_summary} articles based on relevance.")
            ranked_articles = ranked_articles[:max_articles_for_summary]
        else:
            logger.info(f"Number of articles ({len(ranked_articles)}) within the limit for summarization.")

        # Step 2.6: Truncate each abstract if it's too large, to avoid context overflow
        max_chars = 2000  # Customize this value
        for article in ranked_articles:
            abs_text = article.get("abstract", "")
            if not isinstance(abs_text, str):
                abs_text = "" if abs_text is None else str(abs_text)
            if len(abs_text) > max_chars:
                abs_text = abs_text[:max_chars] + "..."
            article["abstract"] = abs_text

        # Update stored articles to the truncated list
        self.latest_pubmed_search['articles'] = ranked_articles
        self.latest_pubmed_search['formatted_articles'] = "\n\n".join([
            f"**Title:** {article['title']}\n"
            f"**Abstract:** {article['abstract']}\n"
            f"**Publication Date:** {article['date']}\n"
            f"**URL:** {article['url']}"
            for article in ranked_articles
        ])
        self.latest_pubmed_search['num_articles'] = len(ranked_articles)

        # Step 3: Generate summary
        summary = self.summarize_articles()

        # Step 4: Compute ROUGE/BLEU metrics
        metrics = self.compute_metrics(summary)

        # Step 5: Judge the summary
        initial_judgment = self.judge_summary(summary)

        # Step 5.5: Follow-up evaluation (ask the judge if it's sure)
        followup_judgment = self.followup_judge(summary, initial_judgment)

        # Return all data
        return {
            "summary": summary,
            "metrics": metrics,
            "judge_evaluation": initial_judgment,
            "followup_judgment": followup_judgment,
            "ranking_metric": ranking_metric,
            "average_similarity_all": average_similarity_all,
            "average_similarity_top": average_similarity_top,
            "agent_prompt": getattr(self, 'agent_prompt', ""),
            "judge_prompt": getattr(self, 'judge_prompt', "")
        }


def run_experiment(agent_keywords, agent_description, model_names, temperatures, top_ps, summary_word_counts, prompting_methods, enable_rankings, topic_id):
    """
    Runs experiments over combinations of model names, temperatures, top-p values, summary word counts,
    prompting methods, and ranking flags for a specific topic, logging detailed initial and follow-up judge evaluations.

    Parameters:
        agent_keywords (list): List of keywords for PubMed search.
        agent_description (str): Description for summary focus.
        model_names (list): List of OpenAI model names to experiment with.
        temperatures (list): List of temperature values to experiment with.
        top_ps (list): List of top-p values to experiment with.
        summary_word_counts (list): List of desired summary word counts to experiment with.
        prompting_methods (list): List of prompting methods to experiment with.
        enable_rankings (list of bool): List containing True and False to toggle ranking.
        topic_id (int): Identifier for the current topic.
    """

    total_runs = len(model_names) * len(temperatures) * len(top_ps) * len(summary_word_counts) * len(prompting_methods) * len(enable_rankings)

    with tqdm(total=total_runs, desc=f"Running experiments for Topic {topic_id}", unit="run") as pbar:
        for idx, (model_name, temperature, top_p, summary_word_count, prompting_method, enable_ranking) in enumerate(itertools.product(
            model_names, temperatures, top_ps, summary_word_counts, prompting_methods, enable_rankings
        ), start=1):

            run_name = f"Topic_{topic_id}_Run_{idx}"

            with mlflow.start_run(run_name=run_name):
                logger.info(f"Starting Run {idx} for Topic {topic_id} with parameters:")

                mlflow.log_param("keywords", ", ".join(agent_keywords))
                mlflow.log_param("framework", "single_agent")
                mlflow.log_param("description", agent_description)
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("temperature", temperature)
                mlflow.log_param("top_p", top_p)
                mlflow.log_param("summary_word_count", summary_word_count)
                mlflow.log_param("prompting_method", prompting_method)
                mlflow.log_param("judge_model", "gpt-3.5-turbo")
                mlflow.log_param("enable_ranking", enable_ranking)

                try:
                    agent = IntelligentAgent(
                        keywords=agent_keywords,
                        description=agent_description,
                        model_name=model_name,
                        temperature=temperature,
                        top_p=top_p,
                        summary_word_count=summary_word_count,
                        prompting_method=prompting_method,
                        enable_ranking=enable_ranking
                    )
                except ValueError as ve:
                    logger.error(f"Initialization Error: {ve}")
                    mlflow.log_param("initialization_error", str(ve))
                    pbar.update(1)
                    continue

                try:
                    results = agent.process()
                except Exception as e:
                    logger.error(f"Processing Error: {e}")
                    mlflow.log_param("processing_error", str(e))
                    pbar.update(1)
                    continue

                if results['metrics']:
                    for metric, score in results['metrics'].items():
                        mlflow.log_metric(metric, score)

                # Log initial judge evaluation
                initial_judge = results.get('judge_evaluation', {})
                if initial_judge:
                    if "error" in initial_judge:
                        mlflow.log_param("judge_initial_error", initial_judge["error"])
                        mlflow.log_text(initial_judge.get('evaluation', 'No evaluation text'), "judge_initial_evaluation.txt")
                    else:
                        for criterion, evaluation in initial_judge.items():
                            if criterion != "Overall Score":
                                mlflow.log_metric(f"Initial_Judge_{criterion}", evaluation.get('score', 0))
                                mlflow.log_param(f"Initial_Judge_{criterion}_Comment", evaluation.get('comment', ''))
                        mlflow.log_metric("Initial_Judge_Overall_Score", initial_judge.get("Overall Score", 0))
                        mlflow.log_text(json.dumps(initial_judge, indent=2), "judge_initial_evaluation.json")

                # Log follow-up judge evaluation
                followup_judge = results.get('followup_judgment', {})
                if followup_judge:
                    if "error" in followup_judge:
                        mlflow.log_param("judge_followup_error", followup_judge["error"])
                        mlflow.log_text(followup_judge.get('evaluation', 'No evaluation text'), "judge_followup_evaluation.txt")
                    else:
                        for criterion, evaluation in followup_judge.items():
                            if criterion != "Overall Score":
                                mlflow.log_metric(f"Followup_Judge_{criterion}", evaluation.get('score', 0))
                                mlflow.log_param(f"Followup_Judge_{criterion}_Comment", evaluation.get('comment', ''))
                        mlflow.log_metric("Followup_Judge_Overall_Score", followup_judge.get("Overall Score", 0))
                        mlflow.log_text(json.dumps(followup_judge, indent=2), "judge_followup_evaluation.json")

                mlflow.log_metric("average_similarity_all_articles", results.get("average_similarity_all", 0.0))
                mlflow.log_metric("average_similarity_top_10_articles", results.get("average_similarity_top", 0.0))
                mlflow.log_text(results['summary'], "summary.txt")

                num_articles = agent.latest_pubmed_search.get("num_articles", 0)
                mlflow.log_param("num_articles", num_articles)

                pbar.update(1)
                time.sleep(2)




if __name__ == "__main__":
    # Define multiple topics from various domains
    topics = [
    {
        "keywords": ["mRNA", "vaccine", "immunology"],
        "description": "Latest advancements and applications of mRNA vaccines in infectious disease immunology."
    },
    {
        "keywords": ["nanoparticle", "drug-delivery", "cancer"],
        "description": "Research on nanoparticles enhancing targeted drug-delivery systems for cancer therapies."
    },
    {
        "keywords": ["microbiome", "gut-health", "probiotics"],
        "description": "Studies exploring gut microbiome's role in health through probiotic interventions."
    },
    {
        "keywords": ["bioinformatics", "genome", "sequencing"],
        "description": "Innovations in bioinformatics for efficient genome sequencing and data analysis methods."
    },
    {
        "keywords": ["stem-cells", "regeneration", "therapy"],
        "description": "Emerging therapies utilizing stem cells for tissue regeneration and medical applications."
    },
    {
        "keywords": ["AI", "protein", "folding"],
        "description": "Advancements using artificial intelligence to accurately predict protein-folding structures."
    },
    {
        "keywords": ["bioplastics", "sustainability", "polymers"],
        "description": "Research on bioplastics and sustainable polymer alternatives reducing environmental impact."
    },
    {
        "keywords": ["gene-therapy", "genetic-disorders", "vectors"],
        "description": "Innovative gene therapy techniques using viral vectors to treat genetic disorders."
    },
    {
        "keywords": ["biosensors", "diagnostics", "point-of-care"],
        "description": "Development of biosensors enhancing rapid diagnostics for point-of-care medical applications."
    },
    {
        "keywords": ["epigenetics", "gene-expression", "disease"],
        "description": "Exploration of epigenetic mechanisms regulating gene-expression changes linked to disease."
    }
]


    # Define parameter ranges
    model_names = ["gpt-3.5-turbo"]  # Ensure these models are supported
    temperatures = [0.0, 0.2, 0.5]  # Low to moderate randomness
    top_ps = [1.0]  # Typical values
    summary_word_counts = [300]  # Desired summary lengths in words
    prompting_methods = [
        "Chain of Thought",
        "Tree of Thought"
    ]
    enable_rankings = [True, False]  # New parameter for ranking toggle

    # Set the experiment name
    mlflow.set_experiment("IntelligentAgent_Optimization_Sum_Ret_v1")

    # Run experiments for each topic
    for idx, topic in enumerate(topics, start=1):

        logger.info(f"Starting experiments for Topic {idx}: {', '.join(topic['keywords'])}")
        run_experiment(
            agent_keywords=topic['keywords'],
            agent_description=topic['description'],
            model_names=model_names,
            temperatures=temperatures,
            top_ps=top_ps,
            summary_word_counts=summary_word_counts,
            prompting_methods=prompting_methods,
            enable_rankings=enable_rankings,
            topic_id=idx  # Pass the topic identifier
        )

    logger.info("All experiments completed. You can view the results using the MLflow UI with `mlflow ui`.")
