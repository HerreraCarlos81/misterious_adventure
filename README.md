# 🚀 Space Exploration Interactive Narrative: A Learning Project

This project demonstrates how to create an interactive narrative experience using large language models (LLMs) and integrate them with a database for storing chat history. The user plays the role of a space traveler exploring a new solar system, while the AI copilot, powered by an LLM, guides the user through challenges, choices, and consequences, dynamically adapting the plot based on the user's choices.

## Learning Objectives 🎒

By implementing this project, you will gain hands-on experience with the following concepts and technologies:

- Utilizing LLMs, such as Anthropic Claude 3 Haiku and Bedrock Mistral 8x7B, for generating contextual responses
- Creating interactive narratives with branching paths and dynamic plot adaptation
- Integrating LLMs with a database (Astra DB) for storing and retrieving chat history
- Handling user input and generating appropriate AI responses based on the conversation flow
- Customizing LLM prompts and templates to guide the AI copilot's behavior and story generation

## LLM Integration 🤖

The project leverages the power of LLMs to generate contextual responses and guide the user through the interactive narrative. The LLMs are integrated using the `langchain` and `langchain_community` libraries, which provide abstractions for working with different LLM providers.

The `BedrockChat` class from `langchain_community` is used to initialize the LLM with the chosen model (Anthropic Claude 3 Haiku or Bedrock Mistral 8x7B). The LLM is then wrapped in an `LLMChain` along with a prompt template and a memory object for storing chat history.

The prompt templates (`claude_template` and `bedrock_template`) define the instructions and rules for the AI copilot, guiding its behavior and response generation. The templates include placeholders for the chat history and user input, allowing the LLM to generate contextually relevant responses.

## Chat History Storage 

To maintain the conversation context and enable the AI copilot to generate responses based on previous interactions, the project integrates with Astra DB for storing chat history. The `AstraDBChatMessageHistory` class from `langchain_astradb` is used to initialize a message history object, which interacts with the Astra DB database.

The chat history is stored in the database using the provided Astra DB endpoint and token. The history is retrieved and passed to the LLM as part of the prompt template, allowing the AI copilot to consider previous conversations when generating responses.

## Dynamic Plot Adaptation 🆕

One of the key features of this project is the dynamic plot adaptation based on user choices. The AI copilot generates a branching narrative experience, where each choice made by the user leads to a new path in the story. The LLM generates appropriate responses and options based on the user's input and the current state of the narrative.

The project demonstrates how to create multiple paths leading to different outcomes, such as success or failure, based on the user's decisions. The AI copilot adapts the plot accordingly and provides relevant choices and consequences at each step of the narrative.

## Customization and Extension

The project provides opportunities for customization and extension. You can modify the prompt templates to adjust the AI copilot's behavior, rules, and story generation. Additionally, you can extend the project by adding more complex branching paths, integrating additional LLMs, or incorporating other features like natural language processing or sentiment analysis.

## Getting Started ⚙️

To get started with this learning project, follow the setup instructions provided in the previous README.md file. Make sure to install the required libraries, set up the necessary API keys, and configure the database connection.

Once the project is set up, you can run the software and interact with the AI copilot to explore the space narrative. Observe how the LLM generates responses based on your choices and how the story adapts dynamically to your decisions.

Feel free to experiment, modify, and extend the project to deepen your understanding of LLMs, interactive narratives, and database integration.

Happy learning and exploring! 🥳