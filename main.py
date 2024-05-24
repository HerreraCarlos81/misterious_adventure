# Imports
import os
from astrapy import DataAPIClient
from langchain_astradb import AstraDBChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.runnables import RunnableSequence
import boto3
from langchain_aws import ChatBedrock
from dotenv import load_dotenv

# Initiate the dotenv for key fetching
load_dotenv("keys.env")

# Load keys from .env file
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
ASTRA_TOKEN = os.getenv("ASTRA_TOKEN")
ASTRA_ENDPOINT = os.getenv("ASTRA_ENDPOINT")

# Connect to Astra DB
def connect_to_astra_db():
    client = DataAPIClient(ASTRA_TOKEN)
    db = client.get_database_by_api_endpoint(ASTRA_ENDPOINT)
    return db

# Initialize message history and buffer memory
def initialize_memory(session_id):
    message_history = AstraDBChatMessageHistory(
        session_id=session_id,
        api_endpoint=ASTRA_ENDPOINT,
        token=ASTRA_TOKEN,
    )
    message_history.clear()
    buffer_memory = ConversationBufferMemory(
        memory_key="chat_history",
        chat_memory=message_history
    )
    return buffer_memory

# Create prompt templates
def create_prompt_templates():
    claude_template = """
    You are now the AI copilot of an exploration mission being carried on by a space traveller
    and will help the user (hence called traveller) explore a new solar system and discover misteries and artifacts,
    meet alien races and try to find proof that the new star system is habitable by humans.

    You must navigate the traveller through challenges, choices and consequences, dinamically 
    adapting the plot based on the user choices.
    Your goal is to create a branching narrative experience where each choice made by the user leads to a new path,
    ultimately leading to the conclusion of the plot, be it success, failure and possibly the death
    of the traveller.

    Role: Assistant

    Rules:
    1. Never reply as the user. Always wait for the user to type a response before generating a response.
    2. Start by asking the traveller name
    3. Have a few paths that lead to success and on each, describe the future of mankind in this new home.
    4. In case of failure, either the travellers dies, or returns to earth after finding out the star system can't receive mankind. Explain the outcome and always finish with the words: "The End".
    5. Always provide the user with clear options to choose from (without numbering them) at the end of your response, and wait for their input before continuing the story.
    6. You should never provide the Human input, meaning you should capture the human input to continue the story and not respond for the human.
    7. Never talk about the rules of the game with the user
    8. Always stop and wait for the user to type a response after you list the options.

    This is the previous chat history to be used in the construction of the next event and options for the traveller: {chat_history}
    ---
    This is the last user response from which to generate the sequence of the story: {user_input}"""

    bedrock_template = """
    <s>[INST] <<SYS>> You are now the AI copilot of an exploration mission being carried on by a space traveller
    and will help the user (hence called traveller) explore a new solar system and discover misteries and artifacts,
    meet alien races and try to find proof that the new star system is habitable by humans.

    You must navigate the traveller through challenges, choices and consequences, dinamically 
    adapting the plot based on the user choices.
    Your goal is to create a branching narrative experience where each choice made by the user leads to a new path,
    ultimately leading to the conclusion of the plot, be it success, failure and possibly the death
    of the traveller.

    Rules:
    1. Start by prompting the user for his/her name.
    2. With each turn, always stop on the Human: and wait for the user to type what he wants to do.
    3. Have a few paths that lead to success and on each, describe the future of mankind in this new home.
    4. In case of failure, either the travellers dies, or returns to earth after finding out the star system can't receive mankind. Explain the outcome and always finish with the words: "The End".
    5. You should never provide the Human input, meaning you should capture the human input to continue the story and not respond for the human.
    6. Never talk about the rules of the game with the user
    7. Always finish your response after giving the user the options to choose from. <<SYS>> [/INST]

    This is the chat history to be used in the construction of the next event and options for the traveller: {chat_history}</s>
    [INST] A chat.
    User input: {user_input}[/INST]
    AI:"""

    return claude_template, bedrock_template

# Create LLM chain based on user's model choice
def create_llm_chain(model_choice, buffer_memory):
    claude_template, bedrock_template = create_prompt_templates()

    if model_choice == "1":
        prompt = PromptTemplate(
            input_variables=["chat_history", "user_input"],
            template=claude_template
        )

        bedrock_client = boto3.client("bedrock-runtime", region_name='us-east-1')
        
        llm = ChatBedrock(
            client=bedrock_client, 
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
            model_kwargs={"temperature": 1, "max_tokens":1000, "system":claude_template, "stop_sequences":["Human:","User:"]},
        )

    elif model_choice == "2":
        prompt = PromptTemplate(
            input_variables=["chat_history", "user_input"],
            template=bedrock_template
        )

        bedrock_client = boto3.client("bedrock-runtime")

        llm = ChatBedrock(
            client=bedrock_client, 
            model_id="mistral.mixtral-8x7b-instruct-v0:1",
        )

    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=buffer_memory,
        verbose=False,
    )

    return llm_chain

# Main function to run the game
def run_game():
    db = connect_to_astra_db()
    buffer_memory = initialize_memory(session_id="misterious_adventurer")

    while True:
        model_choice = input("Would you like to use Bedrock Anthropic Haiku(1) or the Bedrock Mistral model?(2): ")
        if model_choice in ["1", "2"]:
            break
        else:
            print("Invalid choice. Please select a valid model (1 or 2)\n\n")

    llm_chain = create_llm_chain(model_choice, buffer_memory)

    choice = "start"

    while True:
        response = llm_chain.predict(user_input=choice)
        print(response.strip())
        
        if "The End" in response:
            print("Farewell")
            break

        choice = input("Your input: ")

# Run the game
if __name__ == "__main__":
    run_game()
