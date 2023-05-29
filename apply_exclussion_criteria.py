""" The "apply_exclusion_criteria.py" script automates the process of conducting literature reviews using OpenAI's GPT language model.
It applies exclusion criteria to a CSV file of research papers, filtering out papers that do not meet the specified criteria.
The exclusion criteria, defined in a JSON file, determine the papers to be excluded.
The script uses the OpenAI API and allows customization of the model and parameters.
The script is provided under the MIT license, offered as-is, without any liability for results or costs incurred.
It is intended to streamline and expedite literature review tasks, providing researchers with efficient and automated analysis of research papers. """
# Author: Roberto Martínez, May 29, 2023

import argparse
import openai
import pandas as pd
from tqdm import tqdm
import json
import os

def main():
    parser = argparse.ArgumentParser(description='Process a CSV of papers through GPT-4 exclusion criteria.')
    parser.add_argument('csvfile', type=str, help='The CSV file containing the papers to be processed.')
    parser.add_argument('jsonfile', type=str, help='The JSON file containing the exclusion criteria.')
    parser.add_argument('apikey', type=str, help='The OpenAI API key.')
    parser.add_argument('model', type=str, nargs='?', default='gpt-3.5-turbo', help='The OpenAI model to be used. Options are gpt-4 and gpt-3.5-turbo')
    parser.add_argument('--temperature', type=float, default=0.5, help='The temperature parameter for the OpenAI API.')
    parser.add_argument('--max_tokens', type=int, default=3, help='The max_tokens parameter for the OpenAI API.')
    
    args = parser.parse_args()
    
    # Set API key
    openai.api_key = args.apikey
    
    # Load the CSV and JSON files
    try:
        df = pd.read_csv(args.csvfile)
        with open(args.jsonfile) as json_file:
            exclusion_criteria = json.load(json_file)
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return
    
    assistantRole = "You are an expert scitific researcher expert in the fields of multimedia, Virtual Reality, Metaverse, eHealth and signal processing."
    
    # Check each paper against each exclusion criterion
    # Add a progress bar indicator using tqdm
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        title = row['Title']
        abstract = row['Abstract']
        
        for criterion in exclusion_criteria:
            prompt = f"The following abstract is from the paper titled '{title}':\n\n{abstract}\n\nDoes this paper title or abstract {criterion['description']}? Please answer with 'Yes' or 'No'."
            response = openai.ChatCompletion.create(
                model=args.model,
                messages=[
                    {'role': "system", "content": assistantRole},
                    {'role': "user", "content": prompt}
                ],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                stop='\n'
            )
            
            # Check if the response is 'Yes' or 'No'
            response_text = response['choices'][0]['message']['content'].strip().lower()
            if response_text not in ['yes', 'no']:
                print(f"Unexpected response for paper '{title}': {response_text}")
                with open("unexpected_responses.txt", "a") as file:
                    file.write(f"Paper Title: {title}\nResponse: {response_text}\n\n")
                    continue
            # Add a new column for this criterion
            df.at[index, criterion['criteriaId']] = response_text == 'yes'
            
    # Create the 'excluded' column
    criteria_columns = [criterion['criteriaId'] for criterion in exclusion_criteria]
    df['excluded'] = df[criteria_columns].apply(lambda row: any(row), axis=1)
            
    # Save the updated DataFrame to a new CSV file
    base, ext = os.path.splitext(args.csvfile)
    df.to_csv(f"{base}_ec_processed{ext}", index=False)
    
if __name__ == "__main__":
    main()

# This script is used from the command line and takes 6 arguments:
# 1. The CSV file containing the papers to be processed.
# 2. The JSON file containing the exclusion criteria.
# 3. The OpenAI API key.
# 4. The OpenAI model to be used. Options are gpt-4, gpt-3.5-turbo, and others available.
# 5. The temperature parameter for the OpenAI API. Default is 0.5.
# 6. The max_tokens parameter for the OpenAI API. Default is 3.

# Example usage:
# py apply_exclussion_criteria.py papers.csv exclusion_criteria.json %OPENAI_API_KEY% gpt-4 --temperature 0.5 --max_tokens 3