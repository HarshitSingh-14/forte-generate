import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
import re
from scholarly import scholarly, ProxyGenerator


def initialize_scholarly():
    """
    Initializes the scholarly module with proxy settings if necessary.
    """
    pg = ProxyGenerator()
    # Optionally, set up a proxy if you're encountering CAPTCHA or blocks
    # Example using a Tor proxy (ensure Tor is running on your machine)
    # pg.Tor_External(tor_cmd='tor')
    # scholarly.use_proxy(pg)
    return

def parse_scholarly_papers(papers, top_n):
    """
    Parses the list of papers retrieved from scholarly and returns a pandas DataFrame.
    """
    data = []
    for paper in papers[:top_n]:
        title = paper.get('bib', {}).get('title', 'N/A')
        authors = ', '.join(paper.get('bib', {}).get('author', []))
        pub_year = paper.get('bib', {}).get('pub_year', 'N/A')
        citations = paper.get('num_citations', 0)
        url = paper.get('pub_url', 'N/A')  # Get the publication URL
        
        data.append({
            'Title': title,
            'Authors': authors,
            'Publication Year': pub_year,
            'Number of Citations': citations,
            'Link': url
        })
    
    df = pd.DataFrame(data)
    
    # Handle missing columns by filling them with None
    expected_columns = ['Title', 'Authors', 'Publication Year', 'Number of Citations', 'Link']
    for col in expected_columns:
        if col not in df.columns:
            df[col] = None
    
    # Reorder columns
    df = df[expected_columns]
    
    return df

def make_clickable(link):
    """Makes a hyperlink in HTML format."""
    if link and link != 'N/A':
        return f'<a href="{link}" target="_blank">View Paper</a>'
    else:
        return 'N/A'

def main():
    st.set_page_config(page_title="Similar Literature Finder", layout="wide")
    
    st.title("🌿 Similar Literature Finder")
    st.markdown("""
    Welcome to the **Methane Mitigation Literature Assistant**! 📚

    - Upload an **Excel (.xlsx)** file containing methane mitigation technologies.
    - Select a technology to retrieve the top relevant research papers.
    """)
    
    # Initialize scholarly
    initialize_scholarly()
    
    # File uploader
    st.sidebar.header("Upload your data")
    uploaded_file = st.sidebar.file_uploader("Upload your Excel file (.xlsx)", type='xlsx')
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            st.sidebar.success("File uploaded successfully.")
            
            with st.expander("📄 View uploaded data"):
                st.dataframe(df)
            
            required_columns = ['Category (Value Chain)', 'Technology Type', 'Technology Name']
            if all(col in df.columns for col in required_columns):
                technology_names = df['Technology Name'].dropna().unique()
                selected_technology = st.sidebar.selectbox("Select a Technology Name:", technology_names)
                
                # Retrieve the corresponding Category and Technology Type
                selected_row = df[df['Technology Name'] == selected_technology].iloc[0]
                selected_category = selected_row.get('Category (Value Chain)', 'N/A')
                selected_tech_type = selected_row.get('Technology Type', 'N/A')
                
                top_n_options = [10, 20, 30]
                top_n = st.sidebar.selectbox("Number of top relevant papers:", top_n_options)
                
                sort_options = ["Publication Year", "Number of Citations"]
                sort_by = st.sidebar.selectbox("Sort results by:", sort_options)
                
                if st.sidebar.button("Retrieve Relevant Papers"):
                    with st.spinner("Fetching papers..."):
                        try:
                            query = f"{selected_technology} [methane mitigation]"
                            st.markdown(f"### 🔍 Search Techology: `{query}`")
                            
                            # st.write(query)
                            # Perform the search
                            search_results = scholarly.search_pubs(query)
                            
                            # Fetch top N papers
                            papers = []
                            for _ in range(top_n * 2):  # Fetch more to account for possible missing data
                                try:
                                    paper = next(search_results)
                                    papers.append(paper)
                                except StopIteration:
                                    break
                            
                            if not papers:
                                st.error("No papers found for the selected technology.")
                                return
                            
                            # Parse papers into DataFrame
                            papers_df = parse_scholarly_papers(papers, top_n)
                            
                            # Remove duplicates based on Title
                            papers_df.drop_duplicates(subset='Title', inplace=True)
                            
                            # Sorting
                            if sort_by == "Number of Citations":
                                papers_df['Number of Citations'] = pd.to_numeric(
                                    papers_df['Number of Citations'], errors='coerce'
                                ).fillna(0).astype(int)
                                papers_df.sort_values(by='Number of Citations', ascending=False, inplace=True)
                            else:
                                papers_df['Publication Year'] = pd.to_numeric(
                                    papers_df['Publication Year'], errors='coerce'
                                ).fillna(0).astype(int)
                                papers_df.sort_values(by='Publication Year', ascending=False, inplace=True)
                            
                            # Apply clickable links
                            papers_df['Link'] = papers_df['Link'].apply(make_clickable)
                            
                            # Set up to render HTML in DataFrame
                            st.subheader(f"Top {len(papers_df)} Relevant Papers for '{selected_technology}':")
                            st.write(
                                papers_df.to_html(escape=False, index=False),
                                unsafe_allow_html=True
                            )
                        except Exception as e:
                            st.error(f"Error fetching papers: {e}")
            else:
                st.error(f"The file must contain these columns: {required_columns}")
        except Exception as e:
            st.error(f"Error reading the file: {e}")
    else:
        st.sidebar.info("Upload a file to start.")
        st.info("Awaiting file upload.")
    
    st.sidebar.markdown("""
    ---
    Developed by Harshit
    """)

if __name__ == "__main__":
    main()
