from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableParallel, RunnablePassthrough, RunnableLambda,
)

load_dotenv()

model = ChatOpenAI(model="gpt-4o", temperature=0)


def demo_passthrough_chain():
    """A chain that demonstrates passthrough functionality."""
    prompt = ChatPromptTemplate.from_template(
        "Original question: {question}\n"
        "Context: {context}\n\n"
        "Answer the question based on the context."
    )

    # similuatee a retrieve operation
    def fake_retriever(input_dict):
        return " LangChain was created by Harrison Chase in 2022."

    chain = (
        RunnableParallel(
            context=RunnableLambda(fake_retriever), question=RunnablePassthrough()
        )
        | RunnableLambda(
            lambda x: {"context": x["context"], "question": x["question"]["question"]}
        )
        | prompt | model | StrOutputParser()
    )

    result = chain.invoke({"question": "Who created LangChain?"})
    print(f"Answer: {result}")
