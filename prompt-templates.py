from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(
    temperature=0.7,
    model='gpt-3.5-turbo-1106'
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI assistant that provides startup ideas on the following subject."),
        ("human", "{input}")
    ]
)

chain = prompt | llm 
response = chain.invoke({"input": "I want to start a business in the field of AI."})
print(response.content)