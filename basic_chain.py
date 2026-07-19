from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

model = ChatOpenAI(model="gpt-4o", temperature=0)


def demo_basic_chain():
    prompt = ChatPromptTemplate.from_template(
        "Summarize the following text in one sentence: {text}"
    )
    parser = StrOutputParser()
    chain = prompt | model | parser
    result = chain.invoke({
        "text": "LangChain is a framework for developing applications powered by language models."
    })
    print(f"Summary: {result}  ")
