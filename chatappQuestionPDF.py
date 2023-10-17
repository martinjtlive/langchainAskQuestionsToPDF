# to reference env variable - openai_api_key 
from dotenv import load_dotenv 
# to split corpus into chunks
from langchain.text_splitter import CharacterTextSplitter
# for converting text chunks into embeddings
from langchain.embeddings.openai import OpenAIEmbeddings
# to  create semantic index that associates embedding and text chunk
# of corpus. This helps with creation of the Knowledge base 
from langchain.vectorstores import FAISS
# for answering questions 
from langchain.chains.question_answering import load_qa_chain
# Open AI wrapper for langchain question answer
from langchain.llms import OpenAI
# to monitor question cost
from langchain.callbacks import get_openai_callback
# For UX of the chat
import streamlit as st
# to load pdf
from PyPDF2 import PdfReader
# to monitor question cost
from langchain.callbacks import get_openai_callback



def main():
    load_dotenv()
    st.set_page_config(page_title='Ask from the corpus')
    st.header('Question the corpus')



    pdf = st.file_uploader("Upload your PDF", type = "pdf")
    # extract text and create corpus
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        corpus = ''
        for page in pdf_reader.pages:
                corpus += page.extract_text()
        
        st.write(f'Corpus length: {len(corpus)}')
        
        # splitting corpus into chunks
        text_splitter = CharacterTextSplitter(separator='\n'
                                        , chunk_size = 1000
                                        , chunk_overlap = 200
                                        , length_function = len
                                        )
        chunks = text_splitter.split_text(corpus)
        st.write(f'Chunks length: {len(chunks)}')
        
        embeddings = OpenAIEmbeddings()
        
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        
        

        # print(knowledge_base.docstore.__dict__) 
        # print(type(knowledge_base.docstore.__dict__)   ) # dict
        # first2pairs = {k: knowledge_base.docstore.__dict__[k] for k in list(knowledge_base.docstore.__dict__)[:2]} #works
        # print(first2pairs)

        # user input
        user_question = st.text_input("Ask your question:")

        if user_question:
             docs = knowledge_base.similarity_search(user_question)


             llm = OpenAI()
             chain = load_qa_chain(llm, chain_type="stuff")
             with get_openai_callback() as cb: # cb - callback
                  response = chain.run(input_documents=docs, question=user_question)
                  print(cb)
                  st.write(response)


if __name__ == '__main__': main()