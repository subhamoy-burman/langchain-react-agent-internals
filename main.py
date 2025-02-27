import os
from dotenv import load_dotenv
from langchain.agents import tool
from langchain.prompts import PromptTemplate
import langchain.tools.render as render_text_description
from langchain_openai import AzureChatOpenAI
load_dotenv()


@tool
def get_text_length(text:str) -> int:
  text = text.strip("'\n").strip('"')
  return len(text)


if __name__ == '__main__':
  print("Hello React Langchain!")
  #print(get_text_length.invoke(input={"text":"Smelly Cat"}))
  tools = [get_text_length]

  template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought:
    """
  
  prompt = PromptTemplate.from_template(template=template).partial(
    tools=render_text_description(tools), tool_names = ", ".join([t.name for t in tools]))
  

  llm = AzureChatOpenAI(
        temperature=0,
        openai_api_key=os.environ['OPENAI_API_KEY'],
        openai_api_version="2024-08-01-preview",  # Specify API version
        azure_endpoint=os.environ['OPENAI_API_BASE'],
        azure_deployment="gpt-4o",
        model="gpt-4o"
    )
