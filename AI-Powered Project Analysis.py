import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Title and description
st.title("AI-Powered Project Analysis")
st.write("Analyze government project data efficiently with AI and Machine Learning. Upload your Excel file to get started!")

# File uploader
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])

# Placeholder for project data
project_data = None

# Function to read and display uploaded file
if uploaded_file is not None:
    project_data = pd.read_excel(uploaded_file)
    st.write("Data Preview:")
    st.write(project_data.head())

    # Display column names
    st.write("Column names:", project_data.columns.tolist())

    # Analysis section
    st.subheader("Project Analysis")
    
    if 'MINISTRY' in project_data.columns:
        # Count projects per ministry
        ministry_counts = project_data["MINISTRY"].value_counts()
        st.bar_chart(ministry_counts)
        st.write("Number of projects per ministry")

        # Display projects and budgets by ministry
        st.subheader("Projects and Budgets by Ministry")
        selected_ministry = st.selectbox("Select a Ministry", project_data['MINISTRY'].unique())
        ministry_projects = project_data[project_data['MINISTRY'] == selected_ministry]
        st.write(ministry_projects[['ITEM_DESCRIPTION', 'AMOUNT']])

        # Check for similar projects
        st.subheader("Similar Projects within Ministry")
        if len(ministry_projects) > 1:
            # Vectorize the project descriptions
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(ministry_projects['ITEM_DESCRIPTION'].fillna(''))

            # Calculate cosine similarity matrix
            cosine_similarities = cosine_similarity(tfidf_matrix)

            # Display similar projects
            similar_projects = []
            for idx, row in enumerate(cosine_similarities):
                similar_indices = row.argsort()[-3:][::-1]  # Get top 3 similar projects (excluding itself)
                for sim_idx in similar_indices:
                    if idx != sim_idx and row[sim_idx] > 0.5:  # Similarity threshold
                        similar_projects.append({
                            'Project 1': ministry_projects.iloc[idx]['ITEM_DESCRIPTION'],
                            'Project 2': ministry_projects.iloc[sim_idx]['ITEM_DESCRIPTION'],
                            'Similarity': row[sim_idx]
                        })

            if similar_projects:
                st.write("Similar Projects Found:")
                st.write(pd.DataFrame(similar_projects))
            else:
                st.write("No similar projects found.")
        else:
            st.write("Not enough projects in this ministry to find similarities.")
    else:
        st.write("The dataset does not have a 'MINISTRY' column.")
else:
    st.write("Please upload a file to analyze.")
