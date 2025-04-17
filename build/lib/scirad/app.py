import streamlit as st
from scirad.models.agent_single import IntelligentAgent
import time
import pandas as pd

# Set page layout to wide
st.set_page_config(layout="wide")

# Sidebar: Top logo
st.sidebar.image("src/scirad/scirad_logo.png", width=300)  # Replace with your top logo file

# Sidebar with instructions and useful links
st.sidebar.title("Welcome to SciRAD")
st.sidebar.markdown(
    """
    This application showcases an Intelligent Agent designed to help you stay up to date on content.

    **How to use:**
    - **Admin Tab:** Enter a research description, generate recommended keywords, edit them as needed, then click **Run Agentic Search** to see full results (input, summary, metrics, judge evaluation, follow-up judge evaluation, search timeframe, and total cost).
    - **User Tab:** Enter your description, generate recommended keywords, edit them as needed, then click **Approve and Get Summary** to see the final summary and the search timeframe.

    **Useful Links:**
    - [GitHub Repository](https://github.com/alkhalifas/scirad)
    - [PyPI Library](https://pypi.org/project/your_pypi_package)
    - [Slide Deck](https://yourwebsite.com/slide_deck)
    """
)

# Sidebar: Spacer to push content to bottom
st.sidebar.markdown("<br>", unsafe_allow_html=True)

# Sidebar: Bottom logo and copyright
st.sidebar.markdown(
    """
    <div style="text-align: center;">
        <a href="https://devnavigator.com" target="_blank">
            A DevNavigator Product
        </a>
        <p style="font-size: 16px;">© 2025 DevNavigator</p>
        <p style="font-size: 16px;">All Rights Reserved</p>
    </div>
    """, unsafe_allow_html=True
)

# Create two tabs: Admin and User
admin_tab, user_tab = st.tabs(["Admin", "User"])

#############################################
# Admin Tab - Development & Full Results
#############################################
with admin_tab:
    st.header("Admin Panel - Development Mode")
    # Input section (always visible)
    description = st.text_area("Enter Research Description", height=70)

    # Generate keywords button
    if st.button("Generate Recommended Keywords", key="admin_keywords"):
        if not description.strip():
            st.error("Please provide a research description.")
        else:
            temp_agent = IntelligentAgent(
                keywords=["dummy", "placeholder", "sample"],
                description=description,
                model_name="gpt-3.5-turbo"
            )
            recommended_keywords = temp_agent.recommend_keywords(description, n=5)
            st.success("Keywords generated!")
            st.write("**Recommended Keywords:**", recommended_keywords)
            # Store keywords in session_state for further use
            st.session_state.admin_recommended = recommended_keywords

    # Editable keywords field (only shows after keywords are generated)
    if "admin_recommended" in st.session_state:
        edited_keywords = st.text_input(
            "Edit Keywords (comma-separated):",
            value=", ".join(st.session_state.admin_recommended)
        )
        # When the user clicks "Run Agentic Search", process the input and display results in two columns.
        if st.button("Run Agentic Search", key="admin_go"):
            final_keywords = [kw.strip() for kw in edited_keywords.split(",") if kw.strip()]
            if len(final_keywords) < 3:
                st.error("Please provide at least three keywords.")
            else:
                agent = IntelligentAgent(
                    keywords=final_keywords,
                    description=description,
                    model_name="gpt-3.5-turbo",
                    summary_word_count=300,
                    prompting_method="Chain of Thought",
                    enable_ranking=True
                )
                with st.spinner("Processing..."):
                    results = agent.process()
                    time.sleep(1)

                # Create two columns for output
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Input & Summary")
                    st.markdown("#### Research Description")
                    st.write(description)
                    st.markdown("#### Keywords")
                    st.write(final_keywords)
                    st.markdown("#### Summary")
                    st.write(results["summary"])
                with col2:
                    st.markdown("#### Metrics")
                    metrics = results.get("metrics", {})
                    if isinstance(metrics, dict) and metrics:
                        metrics_data = []
                        for key, value in metrics.items():
                            metrics_data.append({"Metric": key, "Value": value})
                        metrics_df = pd.DataFrame(metrics_data)
                        st.table(metrics_df)
                    else:
                        st.write(metrics)

                    st.markdown("#### Judge Evaluation (Initial)")
                    judge_eval = results.get("judge_evaluation", {})
                    if isinstance(judge_eval, dict) and judge_eval:
                        table_data = []
                        for criterion, eval_data in judge_eval.items():
                            if isinstance(eval_data, dict):
                                table_data.append({
                                    "Criterion": criterion,
                                    "Score": eval_data.get("score", ""),
                                    "Comment": eval_data.get("comment", "")
                                })
                            else:
                                table_data.append({
                                    "Criterion": criterion,
                                    "Score": eval_data,
                                    "Comment": ""
                                })
                        judge_df = pd.DataFrame(table_data)
                        custom_css = """
                        <style>
                        .stDataFrame table { 
                          table-layout: fixed; 
                          width: 100% !important;
                        }
                        .stDataFrame th, .stDataFrame td {
                          overflow: hidden;
                          text-overflow: ellipsis;
                          white-space: nowrap;
                        }
                        .stDataFrame th:nth-child(1), .stDataFrame td:nth-child(1) {
                          width: 250px;
                        }
                        .stDataFrame th:nth-child(2), .stDataFrame td:nth-child(2) {
                          width: 250px;
                        }
                        </style>
                        """
                        st.markdown(custom_css, unsafe_allow_html=True)
                        st.dataframe(judge_df, height=300)
                    else:
                        st.write(judge_eval)

                    st.markdown("#### Judge Evaluation (Follow-up)")
                    followup_eval = results.get("followup_judgment", {})
                    if isinstance(followup_eval, dict) and followup_eval:
                        followup_data = []
                        for criterion, eval_data in followup_eval.items():
                            if isinstance(eval_data, dict):
                                followup_data.append({
                                    "Criterion": criterion,
                                    "Score": eval_data.get("score", ""),
                                    "Comment": eval_data.get("comment", "")
                                })
                            else:
                                followup_data.append({
                                    "Criterion": criterion,
                                    "Score": eval_data,
                                    "Comment": ""
                                })
                        followup_df = pd.DataFrame(followup_data)
                        st.dataframe(followup_df, height=300)
                    else:
                        st.write(followup_eval)

                    timeframe = agent.latest_pubmed_search.get("days_back", "N/A")
                    st.markdown(f"**Timeframe Searched:** Last {timeframe} days")
                    if hasattr(agent, "last_cost"):
                        st.markdown(f"**Total Cost:** ${agent.last_cost:.6f}")
                    else:
                        st.markdown("**Total Cost:** N/A")

#############################################
# User Tab - Final Summary Only
#############################################
with user_tab:
    st.header("User Panel")
    user_description = st.text_area("Enter Research Description", height=150, key="user_desc")

    if st.button("Generate Recommended Keywords", key="user_gen"):
        if not user_description.strip():
            st.error("Please provide a research description.")
        else:
            temp_agent_user = IntelligentAgent(
                keywords=["dummy", "placeholder", "sample"],
                description=user_description,
                model_name="gpt-3.5-turbo"
            )
            user_recommended_keywords = temp_agent_user.recommend_keywords(user_description, n=5)
            st.success("Keywords generated!")
            st.write("**Recommended Keywords:**", user_recommended_keywords)
            st.session_state.user_recommended = user_recommended_keywords

    if "user_recommended" in st.session_state:
        user_edited_keywords = st.text_input(
            "Edit Keywords (comma-separated):",
            value=", ".join(st.session_state.user_recommended),
            key="user_edit"
        )
        if st.button("Approve and Get Summary", key="user_go"):
            final_user_keywords = [kw.strip() for kw in user_edited_keywords.split(",") if kw.strip()]
            if len(final_user_keywords) < 3:
                st.error("Please provide at least three keywords.")
            else:
                agent_user = IntelligentAgent(
                    keywords=final_user_keywords,
                    description=user_description,
                    model_name="gpt-3.5-turbo",
                    summary_word_count=300,
                    prompting_method="Chain of Thought",
                    enable_ranking=True
                )
                with st.spinner("Generating Summary..."):
                    user_results = agent_user.process()
                    time.sleep(1)
                st.subheader("Final Summary")
                st.write(user_results["summary"])
                timeframe = agent_user.latest_pubmed_search.get("days_back", "N/A")
                st.markdown(f"**Timeframe Searched:** Last {timeframe} days")
