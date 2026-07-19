from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

model = ChatOpenAI(model="gpt-4o", temperature=0)


def demo_chain_branching():
    """Classify, then pick a chain with a normal if/else."""
    parser = StrOutputParser()
    code = ChatPromptTemplate.from_template("You are a coding expert. Help with: {input}") | model | parser
    general = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Answer: {input}"
    ) | model | parser
    classify = ChatPromptTemplate.from_template(
        "Classify this as 'code' or 'general': {input}\nReturn only the classification."
    ) | model | parser

    for q in ["How do I write a for loop in Python?", "What's the weather like today?"]:
        topic = classify.invoke({"input": q})
        result = (code if "code" in topic.lower() else general).invoke({"input": q})
        print(f"Q: {q}")
        print(f"A: {result[:100]}...\n")
