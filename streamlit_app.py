import streamlit as st
from phi.assistant import Assistant
from phi.llm.groq import Groq
import numpy as np
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import os
import pandas as pd
from supabase import create_client, Client


# Load environment variables
load_dotenv()

# Initialize Supabase client
@st.cache_resource
def init_supabase():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    return create_client(url, key)

# Initialize the assistant
@st.cache_resource
def get_assistant():
    return Assistant(
        llm=Groq(model="llama-3.1-70b-versatile", api_key=os.getenv("GROQ_API_KEY")),
        description="I am a helpful AI assistant powered by Groq. How can I assist you today?",
    )

# Function to fetch data from Supabase
@st.cache_data
def fetch_trainer_data():
    supabase = init_supabase()
    try:
        response = supabase.table("persons").select("*").execute()
        
        # Check if the response contains data
        if response.data:
            return [list(person.values()) for person in response.data]
        else:
            st.error("No data found.")
            return []
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return []

# Function to generate embeddings using SentenceTransformer
@st.cache_resource
def generate_embeddings(data):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = []
    ids = []
    
    for record in data:
        if len(record) < 6:
            st.warning(f"Record {record} doesn't have enough fields, skipping it.")
            continue
        
        try:
            person_id, name, domain, skills, experience, link = record
            description = f"{name}, {experience} years of experience, skilled in {skills}, and works in {domain}."
            embedding = model.encode(description)
            ids.append(person_id)
            embeddings.append(embedding)
        except Exception as e:
            st.error(f"Error processing record {record}: {e}")
    
    return ids, np.array(embeddings)

# Function to build FAISS index
@st.cache_resource
def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# Function to perform FAISS search
def search_in_faiss(index, query_embedding, top_k=3):
    distances, indices = index.search(query_embedding, top_k)
    return distances, indices

# Function to create a clickable button for each trainer
def create_trainer_button(trainer_name, link):
    st.markdown(f'<a href="{link}" target="_blank"><button style="background-color:#4CAF50; color:white; padding:10px; border:none; border-radius:4px; cursor:pointer;">Chat with {trainer_name}</button></a>', unsafe_allow_html=True)

# Streamlit app
st.title("Chat with Groq-powered Assistant")

# Input text box for user question
user_input = st.text_input("What do you want to know?", "")

# Generate embedding for the user query outside the button click logic
model = SentenceTransformer('all-MiniLM-L6-v2')
query_embedding = model.encode([user_input]) if user_input.strip() else None

# Button to get the response
if st.button("Ask"):
    if user_input.strip():
        with st.spinner("Generating response..."):
            # Get the response from Groq-powered assistant
            response_generator = get_assistant().chat(user_input)
            response = "".join([chunk for chunk in response_generator if isinstance(chunk, str)])
            st.markdown(response)

            # Fetch trainer data from Supabase
            trainer_data = fetch_trainer_data()

            # Generate embeddings from CSV data
            ids, embeddings = generate_embeddings(trainer_data)

            # Build FAISS index with embeddings
            faiss_index = build_faiss_index(embeddings)

            # Perform FAISS search for similar results
            distances, indices = search_in_faiss(faiss_index, np.array(query_embedding))

            st.markdown("---")
            st.markdown("### Related Trainers:")
            for idx in indices[0]:
                # Fetch trainer details including link from the result
                trainer = trainer_data[idx]
                if len(trainer) < 6:
                    st.warning(f"Trainer data at index {idx} is incomplete. Skipping.")
                    continue
                
                person_id, name, domain, skills, experience, link = trainer
                
                # Display trainer info in a structured format
                st.markdown(f"""
                *Name:* {name}\n
                *Domain:* {domain}\n
                *Skills:* {skills}\n
                *Experience:* {experience} years\n
                """)
                
                # Create clickable button to chat with trainer
                create_trainer_button(name, link)

                # Add a separator between trainers
                st.markdown("---")
    else:
        st.warning("Please enter a question.")
