#!/opt/homebrew/bin/python3


from crewai import Crew, Process, Agent, Task
from langchain_community.llms import Ollama
from crewai_tools import PDFSearchTool
import os

# Set environment variable for API key
os.environ["OPENAI_API_KEY"] = "NA"

# Configure the LLM using Ollama
llm = Ollama(
    model="llama3",
    base_url="http://localhost:11434"
)



tool = PDFSearchTool(
    pdf='',
    config=dict(
        llm=dict(
            provider="ollama",
            config=dict(
                model="llama3",
                temperature=0.5,
                top_p=1.0,
                stream=False
            ),
        ),
        embedder=dict(
            provider="ollama",
            config=dict(
                model="nomic-embed-text"
            ),
        ),
    )
)




FinAidOfficer = Agent(
    role="Federal Financial Aid Officer",
    goal="To clearly summay PLUS correspondence notes to graduate students",
    backstory="You are the person who acts as a translator to financially illiterate graduate students who take out thousands of dollars in student loans. ",
    llm=llm,
    tools=[tool]
)

summarize = Task(description='Summarize all PDFs',
                      agent= FinAidOfficer, expected_output='The expected output should be 5-8 sentences')



# Assemble a crew with planning enabled
crew = Crew(
    agents=[FinAidOfficer],
    tasks=[summarize],
    verbose=True,  # Enable planning feature
)

# Execute tasks
crew.kickoff()
