import pandas as pd
import json
import re
import openai
from dotenv import load_dotenv
import os
import streamlit as st
import time
import logging


def createValidationData():
    # Load the dataset
    csvfile = r".\datasets\test_dataset.csv"
    df = pd.read_csv(csvfile)
    df = df.dropna()    ## Added this line so that the dataframe does not contain any null values, otherwise it will throw an error in OpenAI fine-tuning
    df_valid = createTrainData(df)
    df_valid["Mood"] = df["Mood"].apply(map_mood)
    return df_valid
    
def loadData(data_file):
    # Load the dataset
    df = pd.read_csv(data_file)
    df = df.dropna()    ## Added this line so that the dataframe does not contain any null values, otherwise it will throw an error in OpenAI fine-tuning
    df_train = createTrainData(df)
    df_train["Mood"] = df["Mood"].apply(map_mood)
    return df_train
    
def createTrainData(df):
    # Create the training data
    df_neu = df[df['Mood']== 0]
    df_pos = df[df['Mood']== 1]
    df_neg = df[df['Mood']== 2]
    
    ## This sampling is based on the distribution of the dataset, required to do this for OpenAI fine-tuning
    df_sample_neu = df_neu.sample(frac=0.01)
    df_sample_pos = df_pos.sample(frac=0.01)
    df_sample_neg = df_neg.sample(frac=0.01)
    
    ## Logging to check the sample sizes are balanced between classes, this can be commented out later on
    logging.error(f"Sample Neutral {df_sample_neu.shape}")
    logging.error(f"Sample Positive {df_sample_pos.shape}")
    logging.error(f"Sample Negative {df_sample_neg.shape}")
    
    ## Combine the samples
    df_sample_sets = pd.concat([df_sample_neu, df_sample_pos, df_sample_neg])
    return df_sample_sets


def map_mood(senti_score):
    if senti_score == 1:
        return "Positive"
    elif senti_score == 2:
        return "Negative"
    else:
        return "Neutral"


## Coverting DF to JSON format
def dfToJSON(df, JSONfile):
    # Convert the DataFrame to the required format
    fine_tune_data = df[['translated_text', 'Mood']].rename(columns={'translated_text': 'prompt', 'Mood': 'completion'})

    # Save the dataset to a JSONL file
    #with open('fine_tune_data.jsonl', 'w') as f:
    with open(JSONfile, 'w') as f:
        for i, row in fine_tune_data.iterrows():
            json.dump({"prompt": row['prompt'], "completion": row['completion']}, f)
            f.write('\n')

    #fine_tune_data.head()

## Fine Tune OpenAI
def fineTune_OpenAI(client, JSONfile):
    # * Upload the training JSON data file for fine tuning
    # * Check the status of the upload job process 
    # * Run the fine tuning based on our training data
    # * Check the status of the fine tuning job process

    ## Upload the training data JON file to OpenAI (we have limit it to 200 for now see the "nrows" when the CSV data is loaded)
    response = client.files.create(file=open(JSONfile, 'rb'),
                                purpose='fine-tune')

    return response.id, response.status

def createJob_OpenAI(client, responseID, valid_respID):
    ## Creata a job to fine-tune the model. It might take a while depending on the size of the file that is being processed
    ## So, ideally we only use a small subset of training data
    finetune_response = client.fine_tuning.jobs.create(
        training_file=responseID,
        validation_file=valid_respID,
        model="davinci-002"
    )
    return finetune_response.id

def checkJob_OpenAI(client, responseID):
    jobStatus = client.fine_tuning.jobs.retrieve(responseID)
   
    ## Once the fine tuning is successful, we will get/used the "fine_tuned_model" from JobStatus object,
    ## otherwise, it will return "None"
    fineTunedModel = jobStatus.fine_tuned_model
    status = jobStatus.status
    return fineTunedModel, status

def listJobEvents(client, responseID):
    ## To list all the events for the fine-tuning job ##
    response = client.fine_tuning.jobs.list_events(responseID, limit=5)
    events = response.data
    events.reverse()

    #for event in events:
    #    print(event.message)
    return events

def getlastModel(client):
    # List 10 fine-tuning jobs
    response = client.fine_tuning.jobs.list(limit=1)
    #response
    for i in enumerate(response):
        #print(f"Job ID: {i[1].id}, Fine tuned Model: {i[1].fine_tuned_model}, Status: {i[1].status}")
        ft_jobid = i[1].id
        ft_model = i[1].fine_tuned_model
        ft_status = i[1].status
        break
    return ft_model, ft_status
    

def AnalyzeSentiment(client, model, tweet):
    response = client.completions.create(
        model=model,
        prompt=tweet,
        max_tokens=1
        #temperature=0,
        #top_p=1,
        #frequency_penalty=0,
        #presence_penalty=0,
        #stop=["\n"]
    )
    return response


def uiTweets():
    user_tweet = st.text_input("Tweet: ")
    return user_tweet


def testTweet(client, fineTunedModel, tweet):
    TestResponse = AnalyzeSentiment(client, fineTunedModel, tweet)

    #print(TestResponse.choices[0].text)
    st.write(TestResponse.choices[0].text)
    

def main():
    # Set up logging configuration
    logging.basicConfig(filename='tweetLLM_errors.log', level=logging.ERROR, 
                    format='%(asctime)s:%(levelname)s:%(message)s')
    
    ## Load your API key from an environment variable or a configuration file
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>")
    client = openai.OpenAI(api_key=openai.api_key)
         
    st.header("NLP with OpenAI : :panda_face:")
     
    user_tweet = st.text_input("Enter Tweet to analyze: ")
    #print(user_tweet)
    #st.write(user_tweet)
    
    button_label = "Evaluate sentiment"
    train_new = 0
    job_status = ""
    opt_rad1 = st.radio("Option", ["Train and fine-tune new model", "Use last Fine-tuned model"], captions = ["Train and fine-tune new model in OpenAI", "Use last Fine-tuned model from OpenAI"])
    if opt_rad1 == "Train and fine-tune new model":
        button_label = "Train new model"
        uploaded_file = st.file_uploader("Choose a file")
        train_new = 1
    else:
        ft_model, ft_status = getlastModel(client)
        job_status = ft_status
        ft_msg = f"Last trained model was *{ft_model}* with status of *{ft_status}*"
        st.write(ft_msg)
   
    
    if st.button(button_label):    
        msg_placeholder = st.empty()
        msg_list = []
        if train_new:    ## Train new model   
            if uploaded_file:     ## This will allow you to upload the datasets to be used to train new model
                dataset = loadData(uploaded_file)
                dataset_v = createValidationData()
                finetune_JSON = "fine_tune_data.jsonl"
                validation_JSON = "validation_data.jsonl"
                dfToJSON(dataset, finetune_JSON)
                dfToJSON(dataset_v, validation_JSON)
                responseID, responseStatus = fineTune_OpenAI(client, finetune_JSON)         ## This line for the training data
                valid_respID, valid_respStatus = fineTune_OpenAI(client, validation_JSON)   ## This line for the validation data
                ##responseID, responseStatus = "001", "In Progress"
                #print(f"Response ID: {responseID}, Status: {responseStatus}")
                #msg1 = f"Response ID: {responseID}, Status : {responseStatus}"
                msg_list.append(f"Response ID: {responseID}, Status : {responseStatus}")
                msg_list.append(f"Response ID: {valid_respID}, Status : {valid_respStatus}")
                msg_placeholder.text_area("Processing", "\n".join(msg_list), height=200)
                responseID = createJob_OpenAI(client, responseID, valid_respID)
                
                with st.spinner("Fine tuning new model in OpenAI..."):
                    while True:
                        fineTunedModel, job_status = checkJob_OpenAI(client, responseID)
                        ##fineTunedModel = "Test tune Model"
                        #msg2 = f"Fine Tune Model : {fineTunedModel} - {type(fineTunedModel)}"
                        msg_list.append(f"Fine Tune Model : {fineTunedModel} - {type(fineTunedModel)}")
                        msg_list.append(f"Job Status : {job_status}")
                        msg_placeholder.text_area("Processing", "\n".join(msg_list), height=200)
                        if job_status == "failed":
                            break
                        else:
                            if fineTunedModel is not None:
                                break
                            else:
                                time.sleep(20)
                                events = listJobEvents(client, responseID)
                                #events = ["Processing...1", "Processing...2", "Processing...3", "Processing...4"]
                                for event in events:
                                    msg_list.append(event.message)
                                    msg_placeholder.text_area("Processing", "\n".join(msg_list), height=200)
                                continue
            else:
                msg_list.append("Training failed! Please upload a file to train new model and try again")
                msg_placeholder.text_area("", "\n".join(msg_list), height=200)
        else:   ## Use the last good model
            st.write(user_tweet)
            if ft_model is not None:
                fineTunedModel = ft_model
        
        if job_status == "failed":
            st.write("Fine-tuning failed. Check your data and please try again.")
        else:
            if job_status == "succeeded":
                TestResponse = AnalyzeSentiment(client, fineTunedModel, user_tweet)
                #TestResponse = "Test response"
                st.write(TestResponse)
                sentiResults = TestResponse.choices[0].text 
                st.write(sentiResults)
    
    #tweet = uiTweets()
    #testTweet(client, fineTunedModel, tweet)
    

if __name__ == "__main__":
    main()