import boto3
import streamlit as st
import os
import uuid
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

## s3_client
s3_client = boto3.client("s3", region_name=os.getenv("AWS_REGION"))
BUCKET_NAME = os.getenv("BUCKET_NAME")
if not BUCKET_NAME:
    raise ValueError("BUCKET_NAME environment variable is not set")

## Bedrock
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

## prompt and chain
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

## import FAISS
from langchain_community.vectorstores import FAISS

bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"
)
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)

folder_path = "/tmp/"

def get_response(llm, faiss_index, question, external_knowledge, word_limit, style_choice):
    """
    Retrieves relevant documents, constructs a prompt based on the selected style,
    and generates a response using the LLM.
    """
    # Step 1: Retrieve relevant documents from the FAISS index
    try:
        docs = faiss_index.similarity_search(question, k=5)  # Adjust 'k' as needed
        context = "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        st.error(f"Error during similarity search: {str(e)}")
        return "An error occurred while retrieving information."

    # Step 2: Create the prompt using the selected style
    prompt_template = get_prompt_template(
        external_knowledge=external_knowledge,
        word_limit=word_limit,
        style_choice=style_choice
    )
    prompt = prompt_template.format(context=context, question=question)

    # Step 3: Generate the response using the LLM
    try:
        response = llm(prompt)
    except Exception as e:
        st.error(f"Error generating response from LLM: {str(e)}")
        return "An error occurred while generating the response."

    return response

def get_unique_id():
    return str(uuid.uuid4())

def load_index(style_key):
    """
    Downloads the FAISS index files from S3 for the given style_key,
    and loads them locally so we can query them.
    
    style_key should be one of ['mail', 'normal', 'report', 'feedback'].
    """
    os.makedirs(folder_path, exist_ok=True)
    
    # Construct the relevant file names based on style_key
    faiss_index_name = f"my_faiss_{style_key}"
    faiss_index_file = faiss_index_name + ".faiss"
    faiss_pkl_file = faiss_index_name + ".pkl"
    
    files_to_download = [faiss_index_file, faiss_pkl_file]
    
    try:
        # Check if the files exist in S3, then download them
        for file_name in files_to_download:
            s3_client.head_object(Bucket=BUCKET_NAME, Key=file_name)
            s3_client.download_file(
                Bucket=BUCKET_NAME, 
                Key=file_name, 
                Filename=os.path.join(folder_path, file_name)
            )
            st.success(f"Successfully downloaded {file_name} to {folder_path}")
    except Exception as e:
        st.error(f"Error downloading files from S3 for style '{style_key}': {str(e)}")
        return None

    # Now try loading the FAISS index from local disk
    try:
        faiss_index = FAISS.load_local(
            index_name=faiss_index_name,
            folder_path=folder_path,
            embeddings=bedrock_embeddings
        )
        return faiss_index
    except Exception as e:
        st.error(f"Error loading FAISS index for style '{style_key}': {str(e)}")
        return None

def get_llm(temperature: float):
    """
    Returns a Bedrock LLM with a large max_tokens_to_sample, 
    so it won't truncate early. We rely on the prompt 
    to enforce the word limit.
    """
    llm = Bedrock(
        model_id="anthropic.claude-v2:1", 
        client=bedrock_client,
        model_kwargs={
            'max_tokens_to_sample': 5000,  # Large enough to avoid premature truncation
            'temperature': temperature
        }
    )
    return llm

def get_prompt_template(external_knowledge: int, word_limit: int, style_choice: str) -> PromptTemplate:
    """
    Returns a PromptTemplate based on style_choice:
      - "Email Style"
      - "Normal Style"
      - "Report Style"
      - "Feedback Style"
    """
    if style_choice == "Email Style":
        # Email style prompt
        prompt_template = f"""
        Human: 
        You have read a set of emails in the context below. These emails have a unique writing style: tone, choice of words, and sentence structure. 
        Please answer and write the mail to the question **in the same writing style** as those emails, signed off as "Shashi". 
        Rely primarily on the given context. If the context doesn't have enough info, 
        you should use up to an external knowledge level of {external_knowledge} (scale 1-10). 
        
        Your response should be of {word_limit} words.

        <context>
        {{context}}
        </context>

        Question: {{question}}

        Assistant (in the same style and within {word_limit} words):
        """
    elif style_choice == "Report Style":
        # Report style prompt
        prompt_template = f"""
        Human:
        You are a professional and knowledgeable AI assistant. The user wants a detailed report like given in all the examples. 
        Please create a well-structured report with the following rules 
        1)
        2)
        3)
        Rely primarily on the given context. If the context doesn't have enough info, you may use up to 
        an external knowledge level of {external_knowledge} (scale 1-10), 
        but if you still cannot find the answer, say "I don't know."

        Limit the response to {word_limit} words.

        <context>
        {{context}}
        </context>

        Question: {{question}}

        Assistant (report style, within {word_limit} words):
        """
    elif style_choice == "Feedback Style":
        # Feedback style prompt
        prompt_template = f"""
        Human:
        The user needs feedback or critique based on the context below having same writing style: tone, choice of words, and sentence structure given for different person( like Feedback given for Person x).
        Please provide the answer to the two questions  
        1)  What is this person's super powers(s
        2) What growth idea(s) do you suggest for this person?
          
        If the context doesn't have enough info, you may use up to an external knowledge.

        Keep your response as less than 60 words(approx 60 words) for each question.

        <context>
        {{context}}
        </context>

        Question: {{question}}

        Assistant (constructive feedback, within 120 words for each question (60 words)):
        """
    else:
        # Normal style prompt
        prompt_template = f"""
        Human:
        You are a helpful and knowledgeable AI assistant. 
        Please answer the question below using the provided context. 
        If the context does not have enough information, 
        you may use up to an external knowledge level of {external_knowledge} (scale 1-10), 
        but if you still cannot find the answer, say "I don't know."
        
        Limit the response to {word_limit} words.

        <context>
        {{context}}
        </context>

        Question: {{question}}

        Assistant (helpful, within {word_limit} words):
        """
    
    return PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )

def main():
    st.title("Email/Report/Normal/Feedback Writer (Style-aware RAG with Word Limit)")

    # UI Controls
    style_choice = st.radio(
        "Select the response style", 
        ("Email Style", "Normal Style", "Report Style", "Feedback Style")
    )
    
    # Map style_choice to the style_key used to load the correct FAISS index
    style_key_map = {
        "Email Style": "mail",
        "Normal Style": "normal",
        "Report Style": "report",
        "Feedback Style": "feedback"
    }
    style_key = style_key_map[style_choice]

    temperature = st.slider("Select Temperature", 0.0, 1.0, 0.5) 
    external_knowledge = st.slider("Select external knowledge level (1 = minimal, 10 = maximum)", 1, 10, 5)
    
    # Instead of max tokens, we use word_limit
    word_limit = st.number_input("Word limit (approximate)", min_value=1, max_value=5000, value=50, step=5)
    
    # Larger, resizable input box for questions
    question = st.text_area(
        "Ask a question or request text",
        height=80,
        placeholder="Enter your question or request here..."
    )

    if st.button("Load Style Index & Generate Styled Response"):
        # 1. Load the relevant index for the chosen style
        st.write(f"Loading index for style: {style_choice}")
        faiss_index = load_index(style_key)
        if not faiss_index:
            st.error("No FAISS index loaded. Please ensure you have run admin.py for this style.")
            return
        
        # 2. Generate response if question is provided
        if question.strip() == "":
            st.warning("Please enter a question or prompt.")
        else:
            with st.spinner("Querying the style-based RAG system..."):
                llm = get_llm(temperature)
                response = get_response(
                    llm, 
                    faiss_index, 
                    question, 
                    external_knowledge, 
                    word_limit, 
                    style_choice
                )
                st.write(response)
                st.success("Done")

if __name__ == "__main__":
    main()
