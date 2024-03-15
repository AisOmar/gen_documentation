"""
Documentation Generator with Guardrails Integration

This script utilizes the OpenAI API through the langchain_openai library to generate documentation for Python functions.
It incorporates Nemoguardrails for applying guardrails to the generated documentation, ensuring the output meets specified standards.
A Gradio interface is provided to allow users to input Python function definitions and receive back their documentation.

Requirements:
- An OpenAI API key set as an environment variable (OPENAI_API_KEY).
- The nemoguardrails, langchain_openai, and gradio packages.

The script sets up a Gradio interface where users can enter Python function definitions and generate documentation based on templates and guardrails configurations.
"""
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
import gradio as gr

api_key = os.environ.get('OPENAI_API_KEY')

# Setup for documentation generation.
documentation_prompt = """
You are a staff software engineer with expertise in Python and always aim to write simple and precise code documentation.
Your code documentation is easy to understand and appreciated by other software engineers.
You will be provided with a function definition below and you have to write the documentation for it.

```python
{input}
"""
documentation_template = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template("You are a helpful AI assistant"),
        HumanMessagePromptTemplate.from_template(documentation_prompt),
    ]
)
output_parser = StrOutputParser()
# Ensure you have the OPENAI_API_KEY set as an environment variable or replace 'os.getenv("OPENAI_API_KEY")' with your key.
llm = ChatOpenAI(openai_api_key=api_key)
documentation_chain = documentation_template | llm | output_parser

# Guardrails setup.
config = RailsConfig.from_path("/content/guardrails/")
guardrails = RunnableRails(config)
chain_with_guardrails = guardrails | documentation_chain

# Gradio interface.
def generate_documentation(functionText):
    documentation = chain_with_guardrails.invoke({'input': functionText})
    return documentation

with gr.Blocks() as demo:
    python_function_text = gr.Textbox(label="Python Function Text")
    generate_documentation_button = gr.Button("Generate Documentation")
    python_function_documentation = gr.Textbox(interactive=True, label="Python Function Documentation")
    generate_documentation_button.click(fn=generate_documentation, inputs=python_function_text, outputs=python_function_documentation)

demo.launch(debug=True, share=True)
