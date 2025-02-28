import os
from typing import Union
from dotenv import load_dotenv
from langchain.agents import tool
from langchain.prompts import PromptTemplate
from langchain.tools.render import render_text_description
from langchain_openai import AzureChatOpenAI
from langchain.agents.output_parsers import ReActSingleInputOutputParser  # Corrected import
from langchain.schema import AgentAction, AgentFinish
from langchain.agents.format_scratchpad import format_log_to_str


load_dotenv()


@tool
def get_text_length(text:str) -> int:
  """
  Calculate the length of the given text after stripping any leading or trailing quotes and newlines.
  """
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
    Thought: {agent_scratchpad}
    """
  
  prompt = PromptTemplate.from_template(template=template).partial(
    tools=render_text_description(tools), tool_names = ", ".join([t.name for t in tools]))
  

  llm = AzureChatOpenAI(
        temperature=0,
        stop=["\nObservation"],
        openai_api_key=os.environ['OPENAI_API_KEY'],
        openai_api_version="2024-08-01-preview",  # Specify API version
        azure_endpoint=os.environ['OPENAI_API_BASE'],
        azure_deployment="gpt-4o",
        model="gpt-4o"
    )
  
  intermediate_steps = []

  
  
  agent = {"input": lambda x:x["input"], 
            "agent_scratchpad":  lambda x: format_log_to_str(x["agent_scratchpad"])
           } | prompt | llm | ReActSingleInputOutputParser()

  agent_step: Union[AgentAction,AgentFinish] = agent.invoke(
     input={"input":"What is the length of the text 'Smelly Cat'?", "agent_scratchpad":intermediate_steps})

  print(agent_step)

  if(isinstance(agent_step, AgentAction)):
    tool_name = agent_step.tool
    def find_tool_by_name(tools, name):
        for tool in tools:
            if tool.name == name:
                return tool
        raise ValueError(f"Tool with name {name} not found")

    tool_to_use = find_tool_by_name(tools, tool_name)
    tool_input = agent_step.tool_input
    observation = tool_to_use.func(str(tool_input))
    print(observation)
    intermediate_steps.append((agent_step, str(observation)))

  agent_step: Union[AgentAction,AgentFinish] = agent.invoke(
     input={"input":"What is the length of the text 'Smelly Cat'?", "agent_scratchpad":intermediate_steps})

  print(agent_step)

  if isinstance(agent_step, AgentFinish):
    print(agent_step.return_values)    