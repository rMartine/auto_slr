""" The "apply_full_text_screening_and_apply_inclusion_criteria.py" script automates the process of conducting literature reviews using OpenAI's GPT language model.
It applies full-text screening criteria to asses the quality of a set of research papers in PDF format.
Simultaneously, it extracts relevant infromation from each papers since the full-text screening process is time-consuming and already using the tokens for the model.

The script uses the OpenAI API and allows customization of the model and parameters.
The script is provided under the MIT license, offered as-is, without any liability for results or costs incurred.
It is intended to streamline and expedite literature review tasks, providing researchers with efficient and automated analysis of research papers. """
########### Author: Roberto Martínez, June 28, 2023 ############

# TO DO:
# 1.- Parametrize the script to assign expertises to the assistant in relevant fields for the review.
# 2.- Split functionality in modules, so the user may manipulate the creation of graphics, the application of the exclusion criteria, and the creation of the exclusion criteria separately.
# 3.- Add a flag column to know if a row has been processed or not. This way, the user may run the script multiple times on the same CSV file, and it will only process the rows that have not been processed yet.

import argparse
import openai
import pandas as pd
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import os
import PyPDF2
from numpy import argsort

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader

def query_openai_model_with_new_question(question, relatedContent, assistantRole, prompt, model, maxTokens=8000, temperature=0.5):
    # Iterate through relatedContent to create a list of messages for the role user like this: {'role': "user", "content": f"The contents of the paper: {relatedContent[i]}"}
    message = [
        {'role': "system", "content": assistantRole},
        {'role': "user", "content": prompt},
        {'role': "user", "content": question},
        {'role': "user", "content": "The contents of the paper related to my question:"}
    ]
    for i in range(len(relatedContent)):
        message.append({'role': "user", "content": f"Entry {i}: {relatedContent[i][0].page_content}"})

    response = openai.ChatCompletion.create(
        model=model,
        messages=message,
        max_tokens=maxTokens,
        temperature=temperature
    )
    return response

def search_vectorstore(query, vectorstore):
    try:
        results = vectorstore.similarity_search_with_score(query)
        return results
    except Exception as e:
        print(e)
        return None

def search_query(query, vectorstore):
    results = search_vectorstore(query, vectorstore)
    return results

def create_vectorstore(chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore, embeddings

def split_document_in_chunks(document, chunkSize=1000, chunkOverlap=0):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunkSize,
        chunk_overlap=chunkOverlap,
        length_function=len,
    )
    chunks = splitter.split_text(document)
    chunks = [chunk.replace("\n", " ") for chunk in chunks]
    return chunks

def load_pdf_to_text(filePath):
    pdfFileObj = open(filePath, 'rb')
    pdfReader = PyPDF2.PdfReader(pdfFileObj)
    fullText = ""
    for pageNum in range(len(pdfReader.pages)):
        page_obj = pdfReader.pages[pageNum]
        pageText = page_obj.extract_text()
        fullText += pageText
    return fullText

def load_and_query_pdf(file_path, query, assistantRole, prompt, maxTokens=150):
    # document = load_pdf(file_path)
    document = load_pdf_to_text(file_path)
    chunks = split_document_in_chunks(document)
    vectorstore, embeddings = create_vectorstore(chunks)
    results = search_query(query, vectorstore)
    answer = query_openai_model_with_new_question(query, results, assistantRole, prompt, maxTokens)
    return answer

def load_config_and_data(baseFolder):
    configPath = os.path.join(baseFolder, 'config.json')
    
    # Check if config file exists
    if not os.path.isfile(configPath):
        raise FileNotFoundError(f"Config file not found at {configPath}")
    
    with open(configPath, 'r') as f:
        config = json.load(f)
        
    # Validate the config
    if not isinstance(config, dict) or \
        'FT_ASSESSMENT_CRITERIA' not in config or \
        'LIST_OF_PAPERS' not in config or \
        'PDFS_FOLDER' not in config:
        raise ValueError("Invalid config file")
    
    criteriaFilePath = os.path.join(baseFolder, config['FT_ASSESSMENT_CRITERIA'])
    papersFilePath = os.path.join(baseFolder, config['LIST_OF_PAPERS'])
    pdfsFolderPath = os.path.join(baseFolder, config['PDFS_FOLDER'])
    
    # Check if criteria JSON file and papers CSV file exists
    if not os.path.isfile(criteriaFilePath):
        raise FileNotFoundError(f"Criteria file not found at {criteriaFilePath}")
    if not os.path.isfile(papersFilePath):
        raise FileNotFoundError(f"Papers file not found at {papersFilePath}")
    
    # Load JSON file
    with open(criteriaFilePath, 'r') as f:
        ftAssessmentCriteria = json.load(f)
        
    # Load CSV file and create copies
    papersList = pd.read_csv(papersFilePath)
    papersListWorking = papersList.copy()
    papersListDone = pd.DataFrame(columns=papersList.columns)
    
    # Check if PDFs folder exists, if not create it
    if not os.path.isdir(pdfsFolderPath):
        os.makedirs(pdfsFolderPath)
    
    return ftAssessmentCriteria, papersList, papersListWorking, papersListDone, pdfsFolderPath


def main():
    parser = argparse.ArgumentParser(description='Process a CSV of papers through GPT-4 inclusion criteria and data extraction.')
    parser.add_argument('basefolder', type=str, help='The base folder where the CSV and JSON files are located.')
    parser.add_argument('csvfile', type=str, help='The CSV file containing the papers to be processed.')
    parser.add_argument('jsonfile', type=str, help='The JSON file containing the quality criteria and the extraction questions.')
    parser.add_argument('apikey', type=str, help='The OpenAI API key.')
    parser.add_argument('model', type=str, nargs='?', default='gpt-3.5-turbo', help='The OpenAI model to be used. Options are gpt-4 and gpt-3.5-turbo')
    parser.add_argument('--temperature', type=float, default=0.5, help='The temperature parameter for the OpenAI API.')
    parser.add_argument('--max_tokens', type=int, default=6000, help='The max_tokens parameter for the OpenAI API.')
    parser.add_argument('--throttle', type=float, default=1, help='The number of seconds to wait between calls to the OpenAI API.')

    args = parser.parse_args()

    typesOfCriterions = {
        "quality": "quality",
        "extraction": "extraction"
    }

    # Use a ternary o

    if args.basefolder != '':
        baseFolder = f"{args.basefolder}/"
    else:
        baseFolder = ""

    #ftAssessmentCriteria, papersList, papersListWorking, papersListDone, pdfs_folder_path = load_config_and_data(baseFolder)
    
    # Set OpenAI API key
    if args.apikey == '':
        openai.api_key = os.environ.get("OPENAI_API_KEY")
    else:
        openai.api_key = args.apikey
    
    # Load the CSV and JSON files
    # If there is a 
    try:
        df = pd.read_csv(args.csvfile)
        with open(args.jsonfile) as json_file:
            extractionCriteria = json.load(json_file)
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return
    
    # Create working c
    
    assistantRole = "You are an expert scitific researcher in the fields of Artificial Inteligence, Aerospace, Data Science, Climate Change, Environmental Sciences, Energy, Electronics, Microgrids, COmputer Science and Electricity."

    
    # Check if the dataset has a column "processesed", if not, add it and assign False to all rows
    if 'processed' not in df.columns:
        df['processed'] = False
    elif df['processed'].all():
        print("All papers have been processed.")
        return
    df = df[df['processed'] == False]
    # Save the rest of the records to join them back after processing the unprocessed ones
    df_rest = df[df['processed'] == True]

    # Check each paper against each inclusion criterion
    # TO DO: Drill down into more detailed error handling and noting it in the dataframe
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        # Check if the row has a title and a file name
        if 'title' in row and 'file_name' in row:
            paperTitle = row['title']
        else:
            continue
        try:
            # Load the paper & create the vectorstore
            filePath = f"{baseFolder}pdfs/{row['file_name']}"
            document = load_pdf_to_text(filePath)
            chunks = split_document_in_chunks(document)
            vectorstore, embeddings = create_vectorstore(chunks)
            
            errorFlag = False
            for criterion in extractionCriteria: # TO DO: Adapt from here
                #print(f"Processing paper '{paperTitle}' with criterion '{criterion['criteriaId']}'")
                prompt = f"I will provide you with the contents of a papr titled '{paperTitle}'. Please read the contents of the paper and answer the following questions:"
                # Try to generate a response from the model if the create method fails, save the response in the unexpected column and move to the next paper
                results = search_query(criterion['description'], vectorstore)
                # use ternary operator to assign max_tokens 3 for "quality" types of criterions and leave the args.max_tokens for "extraction" types of criterions
                maxTokens = args.max_tokens
                # maxTokens = 3 if criterion['type'] == typesOfCriterions['quality'] else args.max_tokens
                # Add to maxTokens the amount of tokens in criterion['description'], assistantRole and prompt
                # Need to use LnagChain OpenAIEmbeddings to get the length of the string
                # # maxTokens += len(criterion['description']) + len(assistantRole) + len(prompt)

                answer = query_openai_model_with_new_question(criterion['description'], results, assistantRole, prompt, args.model, maxTokens, args.temperature)
                if answer is None:
                    df.at[index, 'unexpected'] = "No answer from the model"
                    errorFlag = True
                    continue
                
                # Append response to the dataframe in the corresponding column
                # print(f"Answer for paper '{paperTitle}' and criterion '{criterion['criteriaId']}': {answer}")

                responseText = answer['choices'][0]['message']['content'].strip().lower()
                
                # Add a new column for this criterion and true or false depending if the string "yes" is in the responseText
                df.at[index, f"{criterion['criteriaId']}_bool"] = "yes" in responseText
                df.at[index, criterion['criteriaId']] = responseText
                df.to_csv(f"{baseFolder}output.csv", index=False)
                # Wait for 1 seconds to avoid hitting the API rate limit
                time.sleep(args.throttle)
            if errorFlag:
                df.at[index, 'processed'] = False
            else:
                df.at[index, 'processed'] = True
            time.sleep(10)
        except Exception as e:
            print(f"Exception for paper '{paperTitle}': {e}")
            df.at[index, 'unexpected'] = str(e)
            df.at[index, 'processed'] = False
            continue

    
    
    # Put back together both datasets stacking them vertically
    df = pd.concat([df, df_rest], ignore_index=True)

    # Use criterions that are of "type" quality to create a pie chart
    qualityCriteria = [criterion for criterion in extractionCriteria if criterion['type'] == typesOfCriterions['quality']]

    # Create the 'included' column
    criteria_columns = [criterion['criteriaId'] for criterion in qualityCriteria]
    df['included'] = df[criteria_columns].apply(lambda row: any(row), axis=1)
            
    # Save the updated DataFrame to a new CSV file
    base, ext = os.path.splitext(args.csvfile)
    df.to_csv(f"{base}_ft_processed{ext}", index=False)

    # Check if directory exists, if exist delete it, if not create it
    subFolderName = f"{base}_ft_processed"
    if os.path.exists(subFolderName):
        os.rmdir(subFolderName)
    os.mkdir(subFolderName)
        
    for criterion in qualityCriteria:
        plt.figure()
        # Check if the criterion column exists, if not, skip it
        if criterion['criteriaId'] not in df.columns:
            continue
        df[criterion['criteriaId']].value_counts().plot(kind='pie', autopct='%1.1f%%')
        plt.title(f"{criterion['criteriaId']}")
        plt.ylabel('')
        plt.savefig(f"{subFolderName}/{criterion['criteriaId']}.png")
        plt.close()
    # Create a pie chart for the 'excluded' column, also store it in the folder with the other pie charts
    plt.figure()
    df['included'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title(f"Included")
    plt.ylabel('')
    plt.savefig(f"{subFolderName}/included.png")
    plt.close()



if __name__ == "__main__":
    main()

# This script runs with the following command:
# Test
# py apply_ft_screening_and_inclusion.py ft_review_may30 ./ft_review_may30/test.csv ./inclusion_criteria_and_data_extraction.json %OPENAI_API_KEY% gpt-4 --temperature 0.5 --max_tokens 6000 --throttle 0.5

# Full
# py apply_ft_screening_and_inclusion.py ft_review_may30 ./ft_review_may30/meta-health_screen_csv_20230604_ec_processed.csv ./inclusion_criteria_and_data_extraction.json %OPENAI_API_KEY% gpt-4 --temperature 0.5 --max_tokens 6000 --throttle 0.5


# py apply_ft_screening_and_inclusion.py ft_review_saddik_sept13_2023 ./ft_review_saddik_sept13_2023/paper_list.csv ./ft_review_saddik_sept13_2023/inclusion_criteria_and_data_extraction.json %OPENAI_API_KEY% gpt-4-32k --temperature 0.5 --max_tokens 32000 --throttle 0.5