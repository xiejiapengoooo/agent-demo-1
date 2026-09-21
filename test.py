import os

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

client = ChatOpenAI(
    model="gpt-5.6-sol",
    base_url="https://www.su8.codes/v1",
    api_key=SecretStr(os.environ["SU8_API_KEY"]),
)

result = client.invoke("你好")
print(result.content)
