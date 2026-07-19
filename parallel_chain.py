from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel

load_dotenv()

model = ChatOpenAI(model="gpt-4o", temperature=0)


def demo_parallel_chain():
    """Run multiple chains in parallel."""
    # define individual chains
    summarize_prompt = ChatPromptTemplate.from_template("Summarize in two sentences: {text}")
    keywords_prompt = ChatPromptTemplate.from_template(
        "Extract 5 keywords in the following text: {text}\nReturn as a comma-separated list."
    )
    sentiment_prompt = ChatPromptTemplate.from_template(
        "What is the sentiment of the following text? {text}"
    )
    parser = StrOutputParser()

    # Parallel execution
    analysis_chain = RunnableParallel(
        summary=summarize_prompt | model | parser,
        keywords=keywords_prompt | model | parser,
        sentiment=sentiment_prompt | model | parser,
    )

    text = """
    The new AI features are absolutely incredible! Users are loving the
    faster response times and improved accuracy. However, some have noted
    that the pricing could be more competitive. Overall, the product
    launch has been a massive success with record-breaking adoption rates.
    """

    results = analysis_chain.invoke({"text": text})
    print("Analysis Results:")
    print("Parallel Analysis Results:")
    print(f"  Summary: {results['summary']}")
    print(f"  Keywords: {results['keywords']}")
    print(f"  Sentiment: {results['sentiment']}")
