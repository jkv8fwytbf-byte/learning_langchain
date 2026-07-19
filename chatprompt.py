from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

#prompt = ChatPromptTemplate.from_template("Tell me a {adjective} joke about {topic}")
#messages = prompt.invoke({"adjective": "funny", "topic": "chickens"})

# multi-message templates
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that translates {input_language} to {output_language}.",
        ),
        ("human", "Translate the following text: {text}"),
    ]
)

messages = prompt.format_messages(
    input_language="English", output_language="French", text="I love programming."
)