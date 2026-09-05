# **GPU Dispatcher (gpudisp) 🚀**

🇷🇺 **Кратко о проекте (RU):**  
gpudisp — это MLOps-инструмент для оркестрации и запуска зоопарка нейросетей (LLMs, Vision, Audio, Embeddings) на одной потребительской видеокарте (например, 16 ГБ VRAM).**Главная особенонсть — умный диспетчер памяти (Swap Manager).** Он динамически загружает и выгружает модели из видеопамяти на основе их приоритета и времени простоя, предотвращая ошибки Out-Of-Memory (OOM). Легкие и частые модели висят в памяти постоянно, а тяжелые LLM загружаются по требованию и выгружаются, если к ним нет запросов. Весь этот процесс скрыт за единым асинхронным OpenAI-совместимым API. 

**gpudisp** is a dynamic GPU resource dispatcher and API gateway. It enables running multiple heavy machine learning models (Text, Vision, Audio) concurrently on a single consumer-grade GPU (e.g., 16GB VRAM) without encountering Out-Of-Memory (OOM) crashes.  
It acts as a dynamic swap manager, intelligently loading and unloading models from VRAM based on priority, idle timeouts, and active requests, while exposing a unified, OpenAI-compatible API via LiteLLM.

## **🧠 Core Logic: VRAM Swap Manager**

The heart of the project is the custom swap\_manager. Since a single 16GB GPU cannot hold all models simultaneously, the manager acts as a traffic controller for VRAM:

> * **Priority-Based Swapping:** \* **High Priority (Pre-warmed):** Lightweight or frequently used models (e.g., jina-clip-v2 for embeddings) are loaded on startup and never unloaded due to idle time.  
>  * **Low Priority (On-Demand):** Heavy models (e.g., Qwen2.5-VL, GigaAM) are loaded when a request arrives. If they remain idle for a configured idle\_timeout\_sec (e.g., 60 seconds), they are automatically unloaded from VRAM.  
> * **VRAM Guard:** Before spinning up a new model, the manager checks the currently available VRAM against the model's estimated\_vram\_mb. If there isn't enough memory, it forcefully suspends inactive low-priority models to make room.  
> * **Zero-Downtime Routing:** Client requests wait seamlessly while the requested model is spun up into VRAM, completely abstracting the hardware limitations from the end user.

## **🏗 Architecture**

```mermaid
graph TD
    Client([Client Request]) --> Traefik[Traefik HTTPS Proxy]
    
    subgraph "gpudisp stack (Docker Compose)"
    Traefik --> Gateway[Public Gateway<br/>FastAPI / Auth / Streaming]
    
    Gateway -- "Standard /v1/chat" --> LiteLLM[LiteLLM Router]
    Gateway -- "/v1/audio & Custom" --> SwapMgr[Swap Manager<br/>VRAM Dispatcher]
    
    LiteLLM -- Proxy --> SwapMgr
    LiteLLM -- Proxy --> LlamaCPP[llama.cpp<br/>Text Embeddings]
    
    SwapMgr -. "Loads/Unloads (Low Priority)" .-> Qwen[Qwen2.5/3-VL]
    SwapMgr -. "Loads/Unloads (Low Priority)" .-> Audio[GigaAM / Pyannote / SpeechBrain]
    SwapMgr -. "Pre-warmed (High Priority)" .-> Jina[Jina CLIP v2]
    end
    
    classDef proxy fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#000;
    classDef manager fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000;
    classDef model fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#000;
    
    class Gateway,LiteLLM proxy;
    class SwapMgr manager;
    class LlamaCPP,Qwen,Audio,Jina model;
```

## **✨ Key Features**

> * **Multi-Modal Support out-of-the-box:** \* *Vision LLMs:* Qwen2.5-VL, Qwen3-VL (GGUF via llama.cpp)  
  * *Audio Analysis:* GigaAM (transcription), Pyannote/SpeechBrain (diarization, speaker embeddings, gender classification)  
  * *Embeddings:* Qwen3 (text), Jina CLIP v2 (images)  
> * **OpenAI-Compatible API:** Wraps multiple disparate backends into standard /v1/chat/completions and /v1/embeddings endpoints using LiteLLM.  
> * **Production-Ready Gateway:** Includes an async API gateway handling authentication, request sanitization, error handling, and response streaming.  
> * **Custom Audio Processing:** Patched torchaudio and huggingface\_hub implementations to ensure stable GigaAM transcription and agglomerative clustering for diarization fallbacks.

## **🛠 Tech Stack**

> * **ML & Inference:** PyTorch, Transformers, llama.cpp, SpeechBrain, Pyannote, scikit-learn.  
> * **Backend:** Python 3.12, FastAPI, Uvicorn, httpx, LiteLLM.  
> * **Infrastructure:** Docker, Docker Compose, NVIDIA Container Toolkit, Traefik, bash scripting.

## **🚀 API Example**

Because the service is fully OpenAI-compatible, you can use standard clients (like curl, openai-python, or LangChain) to interact with it:  
```bash
curl [https://gpudisp.example.com/v1/chat/completions](https://gpudisp.example.com/v1/chat/completions) \
  -H "Authorization: Bearer YOUR_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-vl",
    "messages": [
      {
        "role": "user",
        "content": "Describe the architecture of this service."
      }
    ]
  }'
```
