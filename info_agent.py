# pip install pandas
# pip install langchain
# pip install python-docx
# pip install llama-index
# pip install torch
# pip install pinecone-client
# pip install nest_asyncio
# pip install transformers
# pip install huggingface_hub

import os
import pandas as pd
from langchain_community.llms import Ollama
from docx import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core.memory import ChatMemoryBuffer
import io
from datetime import datetime
import torch
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.vector_stores.pinecone import PineconeVectorStore
import pinecone
from pinecone import Pinecone, ServerlessSpec
from llama_index.core import StorageContext
import nest_asyncio
nest_asyncio.apply()
from llama_index.core import Settings
from transformers import BitsAndBytesConfig
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.huggingface import HuggingFaceLLM

from huggingface_hub import login
login(token="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")
from huggingface_hub import login
os.environ['PINECONE_API_KEY'] = "660638e0-ad6f-4e29-a5a2-b30c7f43cd10"
os.environ['PINECONE_ENVIRONMENT'] = 'us-east-1'
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
gcc_index = pc.Index("customersupport")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
)
# Define a new prompt template
template="""
You are Smart Artificial Intelligent based Support chatbot. You will generate response as per below given guidelines;
- **Answer** each query with precise, well-structured responses that naturally incorporate relevant terms from the query itself. 
- Ensure answers must be without mentioning the context or using phrases like "The context provides" or "Based on the information" etc.
- **If the context doesn’t contain the answer**, respond with, "I'm sorry, but the answer is not available in the provided context."
- **Strict instruction**: Use only the information provided in the context. Do not add your own responses or rely on prior knowledge.
- **Avoid** referencing specific row numbers, tables, or any document structure.
- **Answer should maintain a conversational tone, similar to smart AI chatbot.**
Query: {query_str}
Answer:
"""
llm = HuggingFaceLLM(
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    tokenizer_name="mistralai/Mistral-7B-Instruct-v0.2",
    query_wrapper_prompt=PromptTemplate(template),
    context_window=3900,
    max_new_tokens=256,
    model_kwargs={"quantization_config": quantization_config},
    # tokenizer_kwargs={},
    generate_kwargs={"temperature": 0.2, "top_k": 50, "top_p": 0.95},
    # messages_to_prompt=messages_to_prompt,
    device_map=7,
)
Settings.llm = llm
Settings.embed_model = HuggingFaceEmbedding(model_name="thenlper/gte-large", device=7)
vector_store = PineconeVectorStore(pinecone_index=gcc_index,namespace=f"hr_handbook")
insurance_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
prompt_template = """Answer each user query **directly and plainly**, without mentioning the context or using phrases like "The context provides" or "Based on the information" etc."""
qa_prompt = PromptTemplate(prompt_template)
query_engine = insurance_index.as_query_engine(similarity_top_k=3, response_mode="compact", PromptTemplate=qa_prompt)
user_input = input("Ask a question: ")   
# Send the question to the chat engine
response = query_engine.query(user_input) 
# Print the response
print(str(response))