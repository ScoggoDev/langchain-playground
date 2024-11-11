from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser, JsonOutputParser
from pydantic import BaseModel, Field

model = ChatOpenAI(
    model="gpt-3.5-turbo-1106",
    temperature=0.7,
)

def call_string_output_parser():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Tell me a joke about the following suject"),
            ("human", "{input}")
        ]
    )

    parser = StrOutputParser()

    chain = prompt | model | parser

    return chain.invoke({"input": "dog"})

def call_list_output_parser():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Gemerate a list of 5 AI companies from the following country. Return the result as a comma separated list"),
            ("human", "{input}")
        ]
    )

    parser = CommaSeparatedListOutputParser()

    chain = prompt | model | parser

    return chain.invoke({"input": "Uruguay"})
    
def call_json_output_parser():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Extract info from the following phrase.\nFormatting Instructions: {format_instructions}"),
            ("human", "{phrase}")
        ]
    )

    class Person(BaseModel):
        name: str = Field(description="The name of the person.")
        age: int = Field(description="The age of the person.")
        nationality: str = Field(description="The nationality of the person, I.E if the country is France, this should be French")
        items: list = Field(description="A list of items")

    parser = JsonOutputParser(pydantic_object=Person)

    chain = prompt | model | parser

    return chain.invoke({
        "phrase": "I met someone named Lucas, he's from Uruguay and is turning 27 next year. He always carries a watch and a backpack",
        "format_instructions": parser.get_format_instructions()
        })

print(call_json_output_parser())