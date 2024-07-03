import os
import time
import PyPDF2
from numpy import argsort
import openai

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader

def load_pdf(filePath):
    loader = PyPDFLoader(filePath)
    document = loader.load()
    return document

def load_pdf_to_text(filePath):
    pdfFileObj = open(filePath, 'rb')
    pdfReader = PyPDF2.PdfReader(pdfFileObj)
    fullText = ""
    for pageNum in range(len(pdfReader.pages)):
        page_obj = pdfReader.pages[pageNum]
        pageText = page_obj.extract_text()
        fullText += pageText
    return fullText

def split_document_in_chunks(document, chunkSize=1000, chunkOverlap=0):
    # Split document string into chunks
    #chunks = str.split(document, "\n")
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunkSize,
        chunk_overlap=chunkOverlap,
        length_function=len,
    )
    chunks = splitter.split_text(document)
    chunks = [chunk.replace("\n", " ") for chunk in chunks]
    #print(len(chunks))
    return chunks

def create_vectorstore(chunks):
    embeddings = OpenAIEmbeddings()
    #vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore, embeddings

def convert_query_to_vector(query, embeddings):
    vector = embeddings.embed(query)
    return vector

def search_vectorstore(query, vectorstore):
    results = vectorstore.similarity_search_with_score(query)
    return results

def get_top_k_results(results, chunks):
    top_k_results = [chunks[result[0]] for result in results]
    return top_k_results

def get_top_k_results_with_scores(results, chunks):
    top_k_results = [(chunks[result[0]], result[1]) for result in results]
    return top_k_results

def load_vectorstore(vectorstorePath):
    vectorstore = FAISS(vectorstorePath)
    return vectorstore

def search_query(query, vectorstore):
    results = search_vectorstore(query, vectorstore)
    return results

def search_query_with_scores(query, vectorstore, embeddings, k=10):
    queryVector = convert_query_to_vector(query, embeddings)
    results = search_vectorstore(queryVector, vectorstore, k)
    results_with_scores = get_top_k_results_with_scores(results, vectorstore.chunks)
    return results_with_scores

def create_new_openai_chat(prompt, assistantRole, maxTokens=150):
    message=[
        {'role': "system", "content": assistantRole},
        {'role': "user", "content": prompt}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=message,
        temperature=0.5
    )
    return response

def query_openai_model_with_new_question(question, relatedContent, assistantRole, prompt, maxTokens=150):
    # Iterate through relatedContent to create a list of messages for the role user like this: {'role': "user", "content": f"The contents of the paper: {relatedContent[i]}"}
    message = [
        {'role': "system", "content": assistantRole},
        {'role': "user", "content": prompt},
        {'role': "user", "content": question},
        {'role': "user", "content": "The contents of the paper related to my question:"}
    ]
    for i in range(len(relatedContent)):
        message.append({'role': "user", "content": f"Entry {i}: {relatedContent[i][0].page_content}"})

    print(message)


    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=message,
        temperature=0.5
    )
    return response

def load_and_query_pdf(file_path, query, assistantRole, prompt, maxTokens=150):
    # document = load_pdf(file_path)
    document = load_pdf_to_text(file_path)
    chunks = split_document_in_chunks(document)
    vectorstore, embeddings = create_vectorstore(chunks)
    results = search_query(query, vectorstore)
    answer = query_openai_model_with_new_question(query, results, assistantRole, prompt, maxTokens)
    return answer


""" def extract_text_from_pdf(file_path):
    try:
        # Set API key
        # get a value for an Environment variable OPENAI_API_KEY
        apiKey = os.environ.get('OPENAI_API_KEY')
        openai.api_key = apiKey
        pdf_file_obj = open(file_path, 'rb')
        pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
        # Assign the role of the assistant
        assistantRole = "You are an expert scitific researcher in the fields of multimedia, Virtual Reality, Metaverse, eHealth and signal processing."
        prompt = f"I will provide you with the contents of a papr titled '{file_path}'. Please read the contents of the paper and answer the following questions:"
        message=[
            {'role': "system", "content": assistantRole},
            {'role': "user", "content": prompt}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=message,
            temperature=0.5
        )
        print(response)
        for page_num in range(len(pdf_reader.pages)):
            page_obj = pdf_reader.pages[page_num]
            pageText = page_obj.extract_text()
            message=[
                {'role': "user", "content": pageText}
            ]
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=message,
                temperature=0.5
            )
            print(response)
            time.sleep(1)
        # Try to generate a response from the model if the create method fails, save the response in the unexpected column and move to the next paper
        q1 = "Using the contents of the paper (the previous inputs were the actual contents of the paper sent in orderly fashion), please answer the following question: What is the main contribution of the paper?"
        message=[
            {'role': "user", "content": q1}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=message,
            temperature=0.5
        )
        pdf_file_obj.close()
        print(response)
        return response
    except Exception as e:
        pdf_file_obj.close()
        print(f"Exception for paper '{file_path}': {e}")
        return """

if __name__ == '__main__':
    openAiKey = os.environ.get('OPENAI_API_KEY')
    openai.api_key = openAiKey
    file_path = 'Healthcare_in_Metaverse.pdf'
    assistantRole = "You are an expert scitific researcher in the fields of multimedia, Virtual Reality, Metaverse, eHealth and signal processing."
    prompt = f"I will provide you with the contents of a papr titled '{file_path}'. Please read the contents of the paper and answer the following questions:"
    question = "What is the main contribution of the paper?"
    answer = load_and_query_pdf(file_path, question, assistantRole, prompt, maxTokens=5000)
    print(answer)
    #load_pdf(file_path)
    #print(extract_text_from_pdf(file_path))


# This script is used from the command prompt with the following syntax:
# python3 extract_text_from_pdf.py <path_to_pdf_file>