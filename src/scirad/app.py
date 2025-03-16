import streamlit as st
from openai_intelligent_agent import IntelligentAgent
from
import time

# Sidebar with instructions and useful links
st.sidebar.title("Tool Instructions")
st.sidebar.markdown(
    """
    This application showcases the Intelligent Agent tool.

    **How to use:**
    - **Admin Tab:** Enter a research description, generate recommended keywords, edit them as needed, then click **Go** to see full results (summary, metrics, and judge evaluation).
    - **User Tab:** Enter your description, view the recommended keywords, and click **Approve and Get Summary** to see the final summary.

    **Useful Links:**
    - [GitHub Repository](https://github.com/your_github_repo)
    - [PyPI Library](https://pypi.org/project/your_pypi_package)
    - [Slide Deck](https://yourwebsite.com/slide_deck)
    """
)

# Create two tabs: Admin and User
admin_tab, user_tab = st.tabs(["Admin", "User"])

#############################################
# Admin Tab - Development & Full Results
#############################################
with admin_tab:
    st.header("Admin Panel - Development Mode")

    # Input research description
    description = st.text_area("Enter Research Description", height=150)

    if st.button("Generate Recommended Keywords"):
        if not description.strip():
            st.error("Please provide a research description.")
        else:
            # Create a temporary agent instance with dummy keywords (required for instantiation)
            temp_agent = IntelligentAgent(
                keywords=["dummy", "placeholder", "sample"],
                description=description,
                model_name="gpt-3.5-turbo"  # Adjust model if needed
            )
            recommended_keywords = temp_agent.recommend_keywords(description, n=5)
            st.success("Keywords generated!")
            st.write("**Recommended Keywords:**", recommended_keywords)

            # Let admin edit keywords
            edited = st.text_input(
                "Edit Keywords (comma-separated):",
                value=", ".join(recommended_keywords)
            )

            if st.button("Go"):
                final_keywords = [kw.strip() for kw in edited.split(",") if kw.strip()]
                if len(final_keywords) < 3:
                    st.error("Please provide at least three keywords.")
                else:
                    # Instantiate the agent with the approved keywords and description
                    agent = IntelligentAgent(
                        keywords=final_keywords,
                        description=description,
                        model_name="gpt-3.5-turbo",  # or another supported model
                        summary_word_count=300,
                        prompting_method="Chain of Thought",
                        enable_ranking=True
                    )
                    with st.spinner("Processing..."):
                        results = agent.process()
                        # Small delay to simulate processing
                        time.sleep(1)

                    st.subheader("Full Results")
                    st.markdown("### Summary")
                    st.write(results["summary"])
                    st.markdown("### Metrics")
                    st.write(results["metrics"])
                    st.markdown("### Judge Evaluation")
                    st.write(results["judge_evaluation"])

#############################################
# User Tab - Final Summary Only
#############################################
with user_tab:
    st.header("User Panel")

    # Input research description
    user_description = st.text_area("Enter Research Description", height=150, key="user_desc")

    if st.button("Generate Recommended Keywords", key="user_keywords"):
        if not user_description.strip():
            st.error("Please provide a research description.")
        else:
            # Create a temporary agent instance for keyword recommendation
            temp_agent_user = IntelligentAgent(
                keywords=["dummy", "placeholder", "sample"],
                description=user_description,
                model_name="gpt-3.5-turbo"
            )
            user_recommended_keywords = temp_agent_user.recommend_keywords(user_description, n=5)
            st.success("Keywords generated!")
            st.write("**Recommended Keywords:**", user_recommended_keywords)

            if st.button("Approve and Get Summary", key="user_go"):
                # Use the recommended keywords without editing in the user flow
                agent_user = IntelligentAgent(
                    keywords=user_recommended_keywords,
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
