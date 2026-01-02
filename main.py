import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agents import Agent, Runner, function_tool
from agents.extensions.models.litellm_model import LitellmModel

# Load environment variables
load_dotenv()

# Configure Gemini model via LiteLLM
gemini_model = LitellmModel(
    model="gemini/gemini-2.0-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
)


# Define tools that the agent can use
@function_tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: The name of the city to get weather for.
    """
    # Mock weather data - replace with real API call
    weather_data = {
        "new york": "72°F, Partly Cloudy",
        "london": "59°F, Rainy",
        "tokyo": "68°F, Clear",
        "paris": "65°F, Sunny",
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")


@function_tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: A mathematical expression to evaluate (e.g., "2 + 2", "10 * 5").
    """
    try:
        # Only allow safe mathematical operations
        allowed_chars = set("0123456789+-*/(). ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


# Create the agent with Gemini model
assistant_agent = Agent(
    name="Assistant",
    instructions="""You are a helpful assistant powered by Google Gemini. You can help with:
- Answering questions
- Getting weather information for cities
- Performing calculations

Be concise and helpful in your responses.""",
    model=gemini_model,
    tools=[get_weather, calculate],
)


# Request/Response models
class ChatRequest(BaseModel):
    message: str
    stream: bool = False


class ChatResponse(BaseModel):
    response: str


# Lifespan for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting FastAPI + OpenAI Agents SDK server...")
    yield
    print("Shutting down server...")


# Initialize FastAPI app
app = FastAPI(
    title="FastAPI + OpenAI Agents SDK",
    description="A minimal example of FastAPI with OpenAI Agents SDK",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "FastAPI + OpenAI Agents SDK",
        "version": "1.0.0",
        "endpoints": {
            "POST /chat": "Send a message to the agent",
            "GET /health": "Health check",
        },
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "model": "gemini-2.0-flash"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message to the agent and get a response."""
    try:
        # Run the agent with the user's message
        result = await Runner.run(assistant_agent, request.message)
        return ChatResponse(response=result.final_output)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Send a message to the agent and stream the response."""

    async def generate():
        try:
            result = Runner.run_streamed(assistant_agent, request.message)
            async for event in result.stream_events():
                # Stream raw response events
                if event.type == "raw_response_event":
                    if hasattr(event.data, "delta") and event.data.delta:
                        yield f"data: {event.data.delta}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    print("Starting server at http://localhost:8000")
    print("API docs available at http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
