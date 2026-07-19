from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch

load_dotenv()

model = ChatOpenAI(model="gpt-4o", temperature=0)


def demo_chain_branching():
    """A chain that demonstrates branching functionality."""
    # Different prompts for different intents
    code_prompt = ChatPromptTemplate.from_template("You are a coding expert. Help with: {input}")
    general_prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Answer: {input}"
    )

    # Classifier
    classifier_prompt = ChatPromptTemplate.from_template(
        "Classify this as 'code' or 'general': {input}\nReturn only the classification."
    )
    classifer = classifier_prompt | model | StrOutputParser()

    # Branching chain  based on classification
    def is_code_question(input_dict):
        classification = classifer.invoke(input_dict)
        return "code" in classification.lower()

    branch = RunnableBranch(
        (is_code_question, code_prompt | model | StrOutputParser()),
        general_prompt | model | StrOutputParser(),  # default branch
    )

    # Test
    questions = ["How do I write a for loop in Python?", "What's the weather like today?"]
    for q in questions:
        result = branch.invoke({"input": q})
        print(f"Q: {q}")
        print(f"A: {result[:100]}...\n")
