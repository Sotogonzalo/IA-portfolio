---
title: "Práctica 15"
date: 2025-11-18
---

# Práctica 15
## Agentes con LangGraph: RAG, Tools y Memoria Conversacional

## Contexto

## Objetivos
- Diseñar un estado de agente (AgentState) para conversaciones multi-turn.
- Construir un agente con LangGraph que: use un modelo de chat (OpenAI) como reasoner, llame tools externas (RAG + otra tool), y mantenga el historial de conversación.
- Integrar RAG como tool reutilizable (retriever + LLM).
- Agregar tools adicionales (p.ej. utilidades, “servicios” dummy).
- Orquestar LLM + tools en un grafo: assistant ↔ tools con bucles.
- Ejecutar conversaciones multi-turn y observar cómo evoluciona el estado.

## Actividades (con tiempos estimados)

## Desarrollo

## Evidencias
- Se adjuntan imagenes desde "resultado-t15-1.png" a "resultado-t15-11.png" en `docs/assets/`

## Reflexión

---

### Parte 0: SetUp

```python
!pip install -U "langgraph>=0.2.0" \
               "langchain>=0.2.11" "langchain-core>=0.2.33" \
               "langchain-community>=0.2.11" "langchain-openai>=0.2.1" \
               "faiss-cpu" "langchain-text-splitters"

import os

os.environ["OPENAI_API_KEY"] = "sk-proj-__COMPLETAR__"
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGSMITH_TRACING"] = "true"
```
Configuramos las keys e instalamos todo lo necesario. No se commiteará las keys auténticas en github por seguridad.

### SetUp LangGraph

```python
from typing_extensions import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END   # START, END
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

llm = ChatOpenAI(model="gpt-5-mini", temperature=0)

def assistant_node(state: AgentState) -> AgentState:
    # TODO: llamar al modelo con todo el historial
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

builder = StateGraph(AgentState)
builder.add_node("assistant", assistant_node)
builder.add_edge(START, "assistant")
builder.add_edge("assistant", END)

graph = builder.compile()

initial_state = {"messages": [HumanMessage(content="Probando mi primer agente LangGraph :)")]}
result = graph.invoke(initial_state)
print(result["messages"][-1].content)
```

#### Resultado: LLM
<!-- ![Tabla comparativa](../assets/resultado-t14-1.png) -->

## Reflexión

#### ¿Qué diferencia hay entre esto y hacer llm.invoke("prompt") directo?
#####
#### ¿Dónde ves explícitamente que hay un estado que viaja por el grafo?
#####

### Parte 1: Estado del agente con memoria “ligera”

```python
from typing import Optional
from typing_extensions import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    summary: Optional[str]

from typing import Optional
from typing_extensions import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    summary: Optional[str]   # p.ej. Optional[str]

# Tip: podés inicializar summary en None en el estado inicial
initial_state = {
    "messages": [],
    "summary": None
}
```

## Reflexión

#### ¿Qué ventaja tiene guardar un summary en vez de todo el historial?
#####
#### ¿Qué información NO deberías guardar en ese resumen por temas de privacidad?
#####


### Parte 2: Construir un RAG “mini”

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Corpus mínimo (podés cambiarlo por algo de tu dominio)
raw_docs = [
    "LangGraph permite orquestar agentes como grafos de estado.",
    "RAG combina recuperación + generación para mejorar grounding.",
    "LangChain y LangGraph se integran con OpenAI, HuggingFace y más."
]

docs = [Document(page_content=t) for t in raw_docs]

# Split en chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Vector store FAISS
emb = OpenAIEmbeddings()
vs = FAISS.from_documents(chunks, embedding=emb)
retriever = vs.as_retriever(search_kwargs={"k": 3})
```

### Tool Rag como función

```python
from langchain_core.tools import tool

@tool
def rag_search(question: str) -> str:
    """
    __COMPLETAR__: descripción de la tool (qué hace, qué devuelve)
    """
    docs = retrieverretriever.vectorstore.similarity_search(
        question,
        k=retriever.search_kwargs.get("k", 4),
    )
    context = "\n\n".join(d.page_content for d in docs)
    if not context:
        return "No se encontró el documento"   # mensaje en caso de no encontrar nada
    return context
```

## Reflexión

#### ¿Qué cambiarías si el corpus fuera mucho más grande?
#####
#### ¿Qué pasaría si devolvés textos muy largos en el context?
##### 


### Parte 3: Otra tool adicional (no RAG)

```python
from datetime import datetime
from langchain_core.tools import tool

FAKE_ORDERS = {
    "ABC123": "En preparación",
    "XYZ999": "Entregado",
}

@tool
def get_order_status(order_id: str) -> str:
    """
    Devuelve el estado de un pedido ficticio dado su ID.
    """
    status = FAKE_ORDERS.get(order_id)
    if status is None:
        return f"No encontré el pedido {order_id}."
    return f"Estado actual del pedido {order_id}: {status}"

@tool
def get_utc_time(_: str = "") -> str:
    """
    Devuelve la hora actual en UTC (formato ISO).
    """
    return datetime.utcnow().isoformat()
```

## Reflexión

#### ¿Qué problema tendría esta tool si la usás en producción real?
##### 
#### ¿Cómo la harías más segura / robusta?
#####


### Parte 4: LLM con tool calling + ToolNode en LangGraph

```python
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage

# 1) Lista de tools
tools = [rag_search, get_order_status, get_utc_time]  # o tus propias tools

# 2) LLM con tools
llm_with_tools = ChatOpenAI(model="gpt-5-mini", temperature=0).bind_tools(tools)

def assistant_node(state: AgentState) -> AgentState:
    """
    Nodo de reasoning: decide si responder directo o llamar tools.
    """
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# 3) Nodo de tools
tool_node = ToolNode(tools)
```

```python
def route_from_assistant(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END
```

### Grafo completo assistant - tools

```python
builder = StateGraph(AgentState)
builder.add_node("assistant", assistant_node)
builder.add_node("tools", tool_node)

builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    route_from_assistant,
    {
        "tools": "tools",
        END: END
    }
)
builder.add_edge("tools", "assistant")

graph = builder.compile()
```

## Reflexión

#### ¿Dónde está ahora el “reasoning”? ¿En qué nodo?
#####
#### ¿Cómo cambiaría el diseño si tuvieras 10 tools en vez de 2–3?
#####


### Parte 5: Conversación multi-turn con el agente

```python
from langchain_core.messages import HumanMessage

state = {
    "messages": [
        HumanMessage(content="Hola, ¿qué es LangGraph en pocas palabras?")
    ],
    "summary": None
}

result = graph.invoke(state)
print("Respuesta 1:", result["messages"][-1].content)
```


```python
from copy import deepcopy

state2 = deepcopy(result)
state2["messages"].append(HumanMessage(content="Usá tu base de conocimiento y decime qué es RAG."))

result2 = graph.invoke(state2)
print("Respuesta 2:", result2["messages"][-1].content)
```


```python
for event in graph.stream(state2, stream_mode="values"):
    msgs = event["messages"]
    print("Último mensaje:", msgs[-1].type, "→", msgs[-1].content if hasattr(msgs[-1], "content") else msgs[-1])
```

## Reflexión

#### ¿Reconocés cuándo el agente está llamando rag_search vs get_order_status?
##### 
#### ¿Qué tipo de prompts le darías al modelo para que use tools “con criterio”?
##### 


### Parte 6: Memoria ligera (summary)

```python
def memory_node(state: AgentState) -> AgentState:
    summary_llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
    # TODO: armar prompt con summary previo + últimos mensajes
    # y devolver {"summary": nuevo_summary}
    __COMPLETAR__

builder.add_node("memory", memory_node)
# Por ejemplo, siempre después de tools:
builder.add_edge("tools", "memory")
builder.add_edge("memory", "assistant")
```

## Reflexión

#### ¿Cómo decidirías cada cuánto actualizar el summary?
##### 
#### ¿Qué tipo de info deberías excluir del summary?
##### 

### Parte 7: Interfaz Gradio para probar el agente

```python
import gradio as gr

def format_chat_history(messages):
    history = []
    last_user = None
    for msg in messages:
        if isinstance(msg, HumanMessage):
            last_user = msg.content
        elif isinstance(msg, AIMessage):
            history.append((last_user or "Usuario", msg.content))
            last_user = None
    return history


def run_agent(input_text: str, state: dict):
    if not state:
        state = {"messages": [], "summary": None}
    state["messages"].append(HumanMessage(content=input_text))
    result = graph.invoke(state)
    last_msg = result["messages"][-1]
    tools_used = (
        ", ".join(call.name for call in last_msg.tool_calls)
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls
        else "Sin tools"
    )
    chat_history = format_chat_history(result["messages"])
    return chat_history, result, tools_used


with gr.Blocks() as ui:
    chatbot = gr.Chatbot(label="Agente LangGraph")
    prompt = gr.Textbox(label="Pregunta", placeholder="Escribí lo que necesites...")
    agent_state = gr.State()
    tools_log = gr.Markdown("Sin tools aún.")
    prompt.submit(
        run_agent,
        [prompt, agent_state],
        [chatbot, agent_state, tools_log],
    )

ui.launch()
```


### Desafío Integrador: Mini-agente de Soporte con RAG + Tool de Estado

```python

```