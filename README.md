# VectorDB — Build a Vector Database from Scratch in Java

A fully working Vector Database built from scratch in Java with a web UI.
Implements HNSW, KD-Tree, and Brute Force search algorithms side-by-side, plus a RAG pipeline powered by a local LLM via Ollama. Built as an educational project to show how production vector databases actually work under the hood.

## What This Project Does

| Feature | Description |
| :--- | :--- |
| **3 Search Algorithms** | HNSW (production-grade), KD-Tree, Brute Force — run all three and compare speed |
| **3 Distance Metrics** | Cosine similarity, Euclidean distance, Manhattan distance |
| **16D Demo Vectors** | 20 pre-loaded semantic vectors across 4 categories (CS, Math, Food, Sports) |
| **Real Document Embedding**| Paste any text → Ollama embeds it with `nomic-embed-text` (768D) |
| **RAG Pipeline** | Ask questions → HNSW retrieves context → local LLM answers |
| **Full REST API** | CRUD endpoints: insert, delete, search, benchmark, hnsw-info |

## How It Works

    Your Text
        │
        ▼
    Ollama (nomic-embed-text)          ← converts text to a 768-dimensional vector
        │
        ▼
    HNSW Index (Java)                  ← indexes the vector in a multilayer graph
        │
        ▼
    Semantic Search                    ← finds nearest neighbors in vector space
        │
        ▼
    Ollama (llama3.2)                  ← reads retrieved chunks, generates an answer
        │
        ▼
    Answer

## Prerequisites

You need **2 things** installed on your machine:
1. **Java Development Kit (JDK) 17 or higher**
2. **Ollama** (runs the local AI models)

## Step-by-Step Setup

### Step 1 — Install Ollama
1. Go to https://ollama.com and download the installer.
2. Open your terminal/command prompt and pull the required models:
   ```bash
   ollama pull nomic-embed-text
   ollama pull llama3.2
### Step 2 — Compile and Run the Java.
Java compiles to bytecode and runs on the JVM. We are using standard Java libraries, so there are no external dependencies to download.

Open your terminal in the directory containing Main.java.

Compile the code:javac Main.java

Run the server:java Main

You should see:
=== VectorDB Engine (Java) ===
http://localhost:8080
20 demo vectors | 16 dims | HNSW+KD-Tree+BruteForce
Ollama: ONLINE

Open your browser and go to: http://localhost:8080

Architecture Deep Dive
BruteForce: 
$O(N \cdot d)$ Exact, baseline.
KDTree: 
$O(\log N)$ Exact, axis-aligned binary space partitioning. 
Fails in high dimensions (curse of dimensionality).
HNSW: 
$O(\log N)$ Approximate, multilayer small-world graph. 
The industry standard for high-dimensional vectors (768D+).

