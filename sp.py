from typing import List
from pydantic import BaseModel, Field
import nest_asyncio

nest_asyncio.apply()

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings, VectorStoreIndex, Document
from llama_index.core.bridge.pydantic import BaseModel, Field
from typing import List, Optional

from init_llm import *
from extraction import *
from common import *
from llama_index.llms.openai_like import OpenAILike

llm = OpenAILike(
    model="gpt-5-mini",
    api_base="https://api.generative.engine.capgemini.com/v1",
    api_key="q9YaSdQMzy5ugRv9etDlz3sCBH2dRSL56BAHU7RB",
    context_window=128000,
    is_chat_model=True,
    is_function_calling_model=False
)

llm = OpenAI(model="gpt-5-mini",
    api_base="https://openai.generative.engine.capgemini.com/v1",
    api_key="q9YaSdQMzy5ugRv9etDlz3sCBH2dRSL56BAHU7RB",
    context_window=128000)
    


logsTop = 50

response = llm.complete("Hello World!")
print(str(response))
