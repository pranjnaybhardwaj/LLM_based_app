import streamlit as st
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import boto3
import json
import botocore
import uvicorn
from threading import Thread

# Initialize FastAPI application
app = FastAPI()

# Neptune Endpoint
neptune_endpoint = "https://db-neptune-2.cluster-"

# Bedrock Configurations
bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')
modelId = "anthropic.claude-3-haiku-"  #  model version
accept = "application/json"
contentType = "application/json"

class PromptRequest(BaseModel):
    prompt: str

@app.post("/query")
async def query_llm_with_knowledge(request: PromptRequest):
    user_prompt = request.prompt

    # Instructions for the LLM
    llm_instructions = (
        "Always refer to the knowledge graph before answering. "
        "Analyze and understand the knowledge graph data to answer questions. "
        "Do not hallucinate "
    )

    # Fetch Data from Neptune
    try:
        query = """
        SELECT ?s ?p ?o
        WHERE {
          ?s ?p ?o
        }
        LIMIT 1000
        """
        headers = {"Content-Type": "application/sparql-query"}
        response = requests.post(neptune_endpoint, data=query, headers=headers)
        response.raise_for_status()
        neptune_data = response.json()  # Parse the response as JSON

        # Extract Results and Format for Prompt
        results = neptune_data.get('results', {}).get('bindings', [])
        formatted_data = "\n".join(
            [f"{record['s']['value']} {record['p']['value']} {record['o']['value']}" for record in results]
        )

        # Combine User Prompt, Knowledge, and Instructions
        prompt_data = (
            f"{llm_instructions}\n\n"
            f"Command: {user_prompt}\n\n"
            f"Knowledge:\n{formatted_data}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching data from Neptune: {str(e)}")

    # Prepare the Bedrock API Call Body
    messages_API_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": int(500 / 0.75),
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_data
                    }
                ]
            }
        ]
    }

    # Invoke the Bedrock Model
    try:
        response = bedrock_runtime.invoke_model_with_response_stream(
            body=json.dumps(messages_API_body),
            modelId=modelId,
            accept=accept,
            contentType=contentType
        )

        # Process the Streamed Response
        stream = response.get('body')
        output = ""

        if stream:
            for event in stream:
                chunk = event.get('chunk')
                if chunk:
                    chunk_obj = json.loads(chunk.get('bytes').decode())
                    if 'delta' in chunk_obj:
                        delta_obj = chunk_obj.get('delta', None)
                        if delta_obj:
                            text = delta_obj.get('text', None)
                            if text:
                                output += text
                            else:
                                break

        return {"response": output.strip()}

    except botocore.exceptions.ClientError as error:
        if error.response['Error']['Code'] == 'AccessDeniedException':
            raise HTTPException(
                status_code=403,
                detail=f"Access denied: {error.response['Error']['Message']}"
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Error interacting with Bedrock: {error}"
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# Streamlit setup
def main():
    st.title("AI Prototype")

    # Take input from user
    user_prompt = st.text_input("Enter your query:")

    if st.button("Get Response"):
        if user_prompt:
            # Make API request to FastAPI backend
            response = requests.post(
                "http://127.0.0.1:8000/query", 
                json={"prompt": user_prompt}
            )

            if response.status_code == 200:
                st.write("Response from LLM:", response.json()['response'])
            else:
                st.error(f"Error: {response.status_code}")
        else:
            st.error("Please enter a query.")

# Run FastAPI in background
def run_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    # Run FastAPI in the background
    thread = Thread(target=run_fastapi)
    thread.daemon = True
    thread.start()

    # Run Streamlit
    main()
