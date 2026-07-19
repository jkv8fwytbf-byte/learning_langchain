from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()


model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# Structured Output
class MovieReview(BaseModel):
    title: str = Field(description="The title of the movie")
    review: str = Field(description="A brief review of the movie")
    rating: int = Field(description="The rating of the movie out of 10")


# Bind the schema to the model
structured_model = model.with_structured_output(MovieReview)

result = structured_model.invoke("Review: Inception is a mind-bending thriller. 9/10")
print(result)  # MovieReview(title='Inception', review='A mind-bending thriller.', rating=9)
