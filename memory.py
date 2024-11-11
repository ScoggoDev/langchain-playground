from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain_community.chat_message_histories.upstash_redis import (
    UpstashRedisChatMessageHistory,
)

UPSTASH_URL = "https://engaged-shark-36828.upstash.io"
UPSTASH_TOKEN = "key"

history = UpstashRedisChatMessageHistory(
    url=UPSTASH_URL,
    token=UPSTASH_TOKEN,
    session_id="chat1",
    ttl=600
)

model = ChatOpenAI(
    model="gpt-3.5-turbo-1106",
    temperature=0.7
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a strict and disciplinarian AI assistant"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ]
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    chat_memory=history
)

chain = LLMChain(
    llm=model,
    prompt=prompt,
    memory=memory,
    verbose=True
)


#msg1 = {
#    "input" : "My name is Lucas"
#}

#response1 = chain.invoke(msg1)
#print(response1)

msg2 = {
    "input" : "What's my name?"
}

response2 = chain.invoke(msg2)
print(response2)