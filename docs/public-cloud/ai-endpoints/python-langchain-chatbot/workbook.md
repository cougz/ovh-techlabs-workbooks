# LangChain Chatbot Workbook - Python Implementation

!!! info "About this workbook"
    This comprehensive workbook will walk you through building multiple AI chatbot variants using Python LangChain framework with OVHcloud AI Endpoints. Follow each step carefully to master chatbot development from basic to advanced implementations.

## Prerequisites

- [ ] OVHcloud account with AI Endpoints access
- [ ] Python 3.12+ installed
- [ ] Basic Python programming knowledge
- [ ] Understanding of virtual environments
- [ ] Linux/macOS/Windows environment

## Understanding the Code

Before we start, let's understand what we'll be building. This workbook covers four different chatbot implementations, each building upon the previous:

!!! example "Source Code Overview"
    
    === "📋 All Files"
        **Repository**: [ovh/public-cloud-examples](https://github.com/ovh/public-cloud-examples/tree/main/ai/ai-endpoints/python-langchain-chatbot)
        
        | File | Purpose | Action |
        |------|---------|--------|
        | `requirements.txt` | Python dependencies | [📥 Download](https://raw.githubusercontent.com/ovh/public-cloud-examples/main/ai/ai-endpoints/python-langchain-chatbot/requirements.txt) |
        | `chatbot.py` | Basic blocking chatbot | [📥 Download](https://raw.githubusercontent.com/ovh/public-cloud-examples/main/ai/ai-endpoints/python-langchain-chatbot/chatbot.py) |
        | `chatbot-streaming.py` | Streaming responses | [📥 Download](https://raw.githubusercontent.com/ovh/public-cloud-examples/main/ai/ai-endpoints/python-langchain-chatbot/chatbot-streaming.py) |
        | `chatbot-memory.py` | Memory-enabled chatbot | [📥 Download](https://raw.githubusercontent.com/ovh/public-cloud-examples/main/ai/ai-endpoints/python-langchain-chatbot/chatbot-memory.py) |
        | `chatbot-streaming-rag.py` | RAG with streaming | [📥 Download](https://raw.githubusercontent.com/ovh/public-cloud-examples/main/ai/ai-endpoints/python-langchain-chatbot/chatbot-streaming-rag.py) |
        | `rag-files/content.txt` | RAG knowledge base | [📥 Download](https://raw.githubusercontent.com/ovh/public-cloud-examples/main/ai/ai-endpoints/python-langchain-chatbot/rag-files/content.txt) |

    === "📖 Code Structure"
        **Chatbot Evolution**:
        
        `chatbot.py`  
        Basic blocking chatbot with single question-answer interaction
        
        `chatbot-streaming.py`  
        Streaming responses for real-time user experience
        
        `chatbot-memory.py`  
        Conversation memory to maintain context across exchanges
        
        `chatbot-streaming-rag.py`  
        Retrieval-Augmented Generation with streaming and knowledge base
        
        ---
        
        **Key Features**:
        
        • **LangChain Framework** integration with OVHcloud AI Endpoints  
        • **Progressive complexity** from basic to RAG implementation  
        • **Streaming capabilities** for improved user experience  
        • **Memory management** for contextual conversations  
        • **Document retrieval** for knowledge-enhanced responses


## Step 1: System Setup and Python Installation

!!! tip "System Requirements"
    This workbook requires Python 3.12+ and pip for optimal LangChain compatibility. Choose your operating system below.

=== "Debian/Ubuntu"
    ```bash
    # Update system
    sudo apt update && sudo apt upgrade -y

    # Install Python 3.12+ and dependencies
    sudo apt install python3 python3-pip python3-venv curl -y

    # Verify Python version (should be 3.12+)
    python3 --version
    ```

=== "CentOS/RHEL"
    ```bash
    # Update system
    sudo yum update -y

    # Install Python and dependencies
    sudo yum install python3 python3-pip python3-venv curl -y

    # Verify installation
    python3 --version
    ```

=== "macOS"
    ```bash
    # Install Homebrew if needed
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

    # Install Python 3.12+
    brew install python@3.12

    # Verify installation
    python3 --version
    ```

=== "Windows"
    ```powershell
    # Download Python from python.org and install
    # Or use Windows Package Manager
    winget install Python.Python.3.12

    # Verify installation
    python --version
    ```

## Step 2: API Access Token

!!! info "API Key Provided"
    Your Personal API Endpoints API key will be provided by your Customer Trainer.

## Step 3: Create Demo Environment

```bash
# Create demo directory
mkdir ~/langchain-chatbot-demo
cd ~/langchain-chatbot-demo

# Create Python virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/macOS
# OR for Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

Expected output:
```
Successfully installed pip-24.x.x
```

## Step 4: Download and Install Dependencies

Download the requirements file and install all necessary LangChain components:

```bash
# Download requirements.txt
curl -o requirements.txt https://raw.githubusercontent.com/ovh/public-cloud-examples/main/ai/ai-endpoints/python-langchain-chatbot/requirements.txt

# Install all dependencies
pip install -r requirements.txt

# Verify LangChain installation
python -c "import langchain; print(f'LangChain version: {langchain.__version__}')"
```

You should see output showing the installed LangChain version and its components.

## Step 5: Configure Environment Variables

Create a configuration file for your OVHcloud AI Endpoints credentials:

```bash
# Create environment configuration file
nano .env
```

Add your OVHcloud configuration:

```bash
# OVHcloud AI Endpoints Configuration
OVH_AI_ENDPOINTS_ACCESS_TOKEN=your_actual_token_here

# Model Configuration
OVH_AI_ENDPOINTS_MODEL_NAME=Mistral-7B-Instruct-v0.3
OVH_AI_ENDPOINTS_MODEL_URL=https://mistral-7b-instruct-v03.endpoints.kepler.ai.cloud.ovh.net/api/openai_compat/v1

# Embedding Model (for RAG)
OVH_AI_ENDPOINTS_EMBEDDING_MODEL_NAME=multilingual-e5-large
```

!!! note "Model URLs"
    Replace the model URLs with your actual endpoints from the [AI Endpoints catalog](https://endpoints.ai.cloud.ovh.net/).

## Step 6: Download Chatbot Files

Download all the chatbot implementation files:

=== "Option 1: Download Individual Files"
    ```bash
    # Download core chatbot files
    curl -o chatbot.py https://raw.githubusercontent.com/ovh/public-cloud-examples/main/ai/ai-endpoints/python-langchain-chatbot/chatbot.py
    curl -o chatbot-streaming.py https://raw.githubusercontent.com/ovh/public-cloud-examples/main/ai/ai-endpoints/python-langchain-chatbot/chatbot-streaming.py
    curl -o chatbot-memory.py https://raw.githubusercontent.com/ovh/public-cloud-examples/main/ai/ai-endpoints/python-langchain-chatbot/chatbot-memory.py
    curl -o chatbot-streaming-rag.py https://raw.githubusercontent.com/ovh/public-cloud-examples/main/ai/ai-endpoints/python-langchain-chatbot/chatbot-streaming-rag.py
    
    # Create RAG directory and download content
    mkdir rag-files
    curl -o rag-files/content.txt https://raw.githubusercontent.com/ovh/public-cloud-examples/main/ai/ai-endpoints/python-langchain-chatbot/rag-files/content.txt
    ```

=== "Option 2: Clone Repository"
    ```bash
    # Clone the repository
    git clone https://github.com/ovh/public-cloud-examples.git
    
    # Copy chatbot files to your demo directory
    cp public-cloud-examples/ai/ai-endpoints/python-langchain-chatbot/*.py .
    cp -r public-cloud-examples/ai/ai-endpoints/python-langchain-chatbot/rag-files .
    
    # Clean up
    rm -rf public-cloud-examples
    ```

## Step 7: Test Basic Chatbot Implementation

Let's start with the simplest chatbot to verify your setup:

```bash
# Load environment variables
set -a && source .env && set +a

# Test the basic chatbot
python chatbot.py
```

Expected output:
```
👤: What is OVHcloud?
🤖: OVHcloud is a global cloud computing company that provides web hosting, dedicated servers, public and private cloud infrastructure, and domain registration services...
```

!!! tip "Understanding the Basic Chatbot"
    The basic chatbot (`chatbot.py`) demonstrates:
    - LangChain integration with OVHcloud AI Endpoints
    - Simple prompt template usage
    - Blocking (synchronous) AI model invocation

## Step 8: Implement Streaming Chatbot

Now let's test the streaming version for real-time responses:

```bash
# Run the streaming chatbot
python chatbot-streaming.py
```

You should see the response appear gradually, word by word, simulating a natural typing experience.

!!! example "Streaming vs Blocking"
    **Blocking**: Waits for complete response before displaying  
    **Streaming**: Displays response incrementally as it's generated

## Step 9: Test Memory-Enabled Chatbot

The memory-enabled chatbot maintains conversation context:

```bash
# Run the memory chatbot
python chatbot-memory.py
```

Expected output:
```
👤: Hello, my name is Stéphane
🤖: Hello Stéphane! Nice to meet you. I'm Nestor, your virtual assistant...

👤: What is my name?
🤖: Your name is Stéphane, as you mentioned when you introduced yourself...
```

!!! tip "Memory Window"
    This implementation uses a memory window of 10 exchanges, keeping the last 10 question-answer pairs for context.

## Step 10: Implement RAG-Enabled Chatbot

The most advanced implementation uses Retrieval-Augmented Generation:

```bash
# Run the RAG chatbot
python chatbot-streaming-rag.py
```

Expected output:
```
👤: Which company created AI Endpoints?
🤖: Based on the provided context, OVHcloud created AI Endpoints. OVHcloud's AI Endpoints platform is designed to provide developers with access to powerful AI models...
```

!!! example "RAG Architecture"
    **Components**:
    - **Document Loader**: Reads content from `rag-files/content.txt`
    - **Text Splitter**: Breaks documents into chunks
    - **Embedding Model**: Creates vector representations
    - **Vector Store**: Stores searchable embeddings (using Chroma)
    - **Retriever**: Finds relevant context for queries

## Step 11: Interactive Chatbot Development

Let's create an interactive version of each chatbot for hands-on testing:

```bash
# Create interactive chatbot script
nano interactive_chatbot.py
```

```python
import os
import sys
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory

# Load environment variables
OVH_AI_ENDPOINTS_ACCESS_TOKEN = os.environ.get('OVH_AI_ENDPOINTS_ACCESS_TOKEN')
OVH_AI_ENDPOINTS_MODEL_NAME = os.environ.get('OVH_AI_ENDPOINTS_MODEL_NAME')
OVH_AI_ENDPOINTS_MODEL_URL = os.environ.get('OVH_AI_ENDPOINTS_MODEL_URL')

if not all([OVH_AI_ENDPOINTS_ACCESS_TOKEN, OVH_AI_ENDPOINTS_MODEL_NAME, OVH_AI_ENDPOINTS_MODEL_URL]):
    print("❌ Missing environment variables. Please check your .env file.")
    sys.exit(1)

# Configure the model
model = ChatMistralAI(
    model=OVH_AI_ENDPOINTS_MODEL_NAME,
    api_key=OVH_AI_ENDPOINTS_ACCESS_TOKEN,
    endpoint=OVH_AI_ENDPOINTS_MODEL_URL,
    max_tokens=1500,
    streaming=True
)

# Setup memory and conversation
memory = ConversationBufferWindowMemory(k=5)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are Nestor, a helpful virtual assistant. The conversation history is: {history}"),
    ("human", "{input}"),
])

conversation = ConversationChain(llm=model, memory=memory, prompt=prompt)

print("🤖 Nestor AI Assistant (powered by OVHcloud AI Endpoints)")
print("Type 'quit' to exit\n")

while True:
    user_input = input("👤 You: ")
    if user_input.lower() in ['quit', 'exit', 'bye']:
        print("🤖 Goodbye! Thanks for chatting with Nestor.")
        break
    
    print("🤖 Nestor: ", end="", flush=True)
    try:
        response = conversation.predict(input=user_input)
        print()  # New line after streaming response
    except Exception as e:
        print(f"❌ Error: {e}")
```

Test the interactive chatbot:

```bash
# Run interactive chatbot
python interactive_chatbot.py
```

## Step 12: Testing Scenarios

### Scenario A: Basic Q&A Testing

Test each chatbot variant with consistent questions:

**Test Questions:**
1. "What is OVHcloud?"
2. "What services does OVHcloud provide?"
3. "How does AI Endpoints work?"

**Run Tests:**
```bash
# Test each implementation
echo "Testing Basic Chatbot..."
python chatbot.py

echo "Testing Streaming Chatbot..."
python chatbot-streaming.py

echo "Testing Memory Chatbot..."
python chatbot-memory.py

echo "Testing RAG Chatbot..."
python chatbot-streaming-rag.py
```

### Scenario B: Memory Persistence Testing

Test the memory chatbot's ability to maintain context:

1. **Introduction**: "Hello, I'm working on a Python project"
2. **Follow-up**: "What programming language am I using?"
3. **Context Test**: "Can you help me with my project?"

### Scenario C: RAG Knowledge Testing

Test the RAG chatbot's ability to use the knowledge base:

1. **Knowledge Query**: "What are the security features of AI Endpoints?"
2. **Specific Details**: "How does OVHcloud protect my data?"
3. **Technical Features**: "What APIs does AI Endpoints support?"

## Expected Results

✅ **Successful Implementation:**

- All four chatbot variants run without errors
- Basic chatbot provides single responses to queries
- Streaming chatbot shows real-time response generation
- Memory chatbot maintains conversation context across exchanges
- RAG chatbot incorporates knowledge base content in responses

✅ **Learning Outcomes:**

- Understanding of LangChain framework architecture
- Experience with different response generation patterns
- Knowledge of memory management in conversations
- Practical implementation of RAG systems
- Integration skills with cloud-based AI services

## Troubleshooting

### Common Issues

**Environment Variable Errors:**
```bash
# Test your environment variables
python -c "import os; print('Token:', 'SET' if os.environ.get('OVH_AI_ENDPOINTS_ACCESS_TOKEN') else 'MISSING')"
```

**LangChain Import Errors:**
```bash
# Reinstall LangChain components
pip uninstall langchain langchain-mistralai langchain-community -y
pip install -r requirements.txt
```

**Model Connection Issues:**
- Verify your model URLs in the AI Endpoints catalog
- Check token permissions and expiration
- Ensure selected models are available in your region

**RAG Implementation Problems:**
- Verify `rag-files/content.txt` exists and contains text
- Check embedding model configuration
- Ensure sufficient memory for vector operations

**Streaming Issues:**
- Check network connectivity and latency
- Verify streaming parameter is set to `True`
- Monitor console output for error messages

## Step 13: Customize Your Knowledge Base

!!! success "Take It Further!"
    Now that you've mastered the basics, it's time to make the RAG chatbot truly your own by adding custom knowledge content.

The RAG chatbot uses the `rag-files/content.txt` file as its knowledge base. Let's customize it with your own domain expertise to see how the chatbot adapts to different types of knowledge.

### Understanding the Current Knowledge Base

First, let's examine what's currently in the knowledge base:

```bash
# View the current content
cat rag-files/content.txt
```

You'll see it contains information about OVHcloud AI Endpoints. Now let's replace this with your own knowledge!

### Customization Ideas

Choose one of these approaches based on your interests:

=== "Technical Documentation"
    **Create a technical knowledge base:**
    
    ```bash
    # Backup original content
    cp rag-files/content.txt rag-files/content-original.txt
    
    # Create new technical content
    cat > rag-files/content.txt << 'EOF'
    Python is a high-level programming language known for its simplicity and readability. It supports multiple programming paradigms including procedural, object-oriented, and functional programming.

    Virtual environments in Python allow you to create isolated Python installations for different projects. You can create a virtual environment using 'python -m venv myenv' and activate it with 'source myenv/bin/activate' on Linux/Mac or 'myenv\Scripts\activate' on Windows.

    LangChain is a framework for developing applications powered by language models. It provides abstractions for working with LLMs, memory management, and tool integration. The framework supports multiple model providers and offers components for building RAG systems.

    Docker is a containerization platform that packages applications and their dependencies into lightweight containers. You can build images using Dockerfiles and run containers with 'docker run'. Docker Compose allows you to define multi-container applications using YAML files.
    EOF
    ```

=== "Business Knowledge"
    **Create a business/company knowledge base:**
    
    ```bash
    # Backup original content
    cp rag-files/content.txt rag-files/content-original.txt
    
    # Create new business content
    cat > rag-files/content.txt << 'EOF'
    Our company offers comprehensive cloud consulting services including infrastructure design, migration planning, and security assessments. We specialize in multi-cloud strategies and hybrid deployments.

    Our pricing model is based on project complexity and duration. Small projects start at $5,000, medium projects range from $15,000-$50,000, and enterprise implementations begin at $100,000. We offer flexible payment terms and maintenance packages.

    Our support team operates 24/7 with guaranteed response times of 1 hour for critical issues, 4 hours for high priority, and 24 hours for standard requests. We provide phone, email, and chat support in multiple languages.

    We follow industry best practices for security including SOC 2 compliance, regular penetration testing, and encrypted communications. All client data is stored in geographically distributed data centers with automatic backups.
    EOF
    ```

=== "Educational Content"
    **Create an educational knowledge base:**
    
    ```bash
    # Backup original content
    cp rag-files/content.txt rag-files/content-original.txt
    
    # Create new educational content
    cat > rag-files/content.txt << 'EOF'
    Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. The three main types are supervised learning, unsupervised learning, and reinforcement learning.

    Supervised learning uses labeled training data to learn a mapping function from inputs to outputs. Common algorithms include linear regression for continuous outputs and logistic regression for classification tasks. Decision trees and random forests are popular for their interpretability.

    Neural networks are computing systems inspired by biological neural networks. They consist of layers of interconnected nodes (neurons) that process information. Deep learning uses neural networks with multiple hidden layers to learn complex patterns in data.

    Natural Language Processing (NLP) enables computers to understand, interpret, and generate human language. Key tasks include tokenization, part-of-speech tagging, named entity recognition, and sentiment analysis. Modern NLP relies heavily on transformer architectures.
    EOF
    ```

=== "Personal Interest"
    **Create content about your hobby or interest:**
    
    ```bash
    # Backup original content
    cp rag-files/content.txt rag-files/content-original.txt
    
    # Example: Cooking knowledge base
    cat > rag-files/content.txt << 'EOF'
    Cooking pasta requires boiling salted water at a ratio of 1 liter per 100g of pasta. Add pasta when water reaches a rolling boil and stir immediately to prevent sticking. Most dried pasta cooks in 8-12 minutes depending on shape and thickness.

    Knife skills are fundamental to cooking. Keep knives sharp for safety and efficiency. The basic cuts include julienne (matchstick), brunoise (small dice), chiffonade (thin strips for herbs), and rough chop. Always use a stable cutting board and proper hand positioning.

    Food safety requires maintaining proper temperatures: refrigerate perishables below 40°F, cook poultry to 165°F, and ground meats to 160°F. Follow the two-hour rule: don't leave food at room temperature for more than two hours, or one hour if ambient temperature exceeds 90°F.

    Seasoning should be done in layers throughout cooking, not just at the end. Salt enhances natural flavors and should be added early. Acids like lemon juice or vinegar brighten flavors and should often be added near the end of cooking to preserve their impact.
    EOF
    ```

### Test Your Custom Knowledge Base

After updating the content, test how your chatbot responds to questions about your new knowledge domain:

```bash
# Test the customized RAG chatbot
python chatbot-streaming-rag.py
```

**Try these test questions based on your chosen content:**

- **Technical**: "How do I create a virtual environment in Python?"
- **Business**: "What are your pricing models for cloud consulting?"
- **Educational**: "What is the difference between supervised and unsupervised learning?"
- **Personal**: "How long should I cook pasta?"

### Advanced Customization Challenges

🎯 **Challenge 1: Multiple Knowledge Sources**
```bash
# Create a more comprehensive knowledge base
cat company_info.txt product_docs.txt faq.txt > rag-files/content.txt
```

🎯 **Challenge 2: Structured Knowledge**
```bash
# Add section headers to organize knowledge
cat > rag-files/content.txt << 'EOF'
=== GETTING STARTED ===
[Your getting started content here]

=== ADVANCED FEATURES ===
[Your advanced content here]

=== TROUBLESHOOTING ===
[Your troubleshooting content here]
EOF
```

🎯 **Challenge 3: Domain-Specific Terminology**
```bash
# Include specialized terms and definitions
cat > rag-files/content.txt << 'EOF'
API (Application Programming Interface): A set of protocols and tools for building software applications...
[Continue with your domain-specific glossary]
EOF
```

### Experimentation Ideas

**Compare Response Quality:**
1. Test the same question with original OVHcloud content
2. Test with your custom content
3. Observe how response style and accuracy change

**Knowledge Gaps Testing:**
1. Ask questions your knowledge base can't answer
2. Notice how the chatbot handles unknown topics
3. Experiment with adding more specific content

**Content Format Testing:**
1. Try bullet points vs. paragraphs
2. Test structured vs. conversational content
3. Experiment with different content lengths

### Share Your Results

!!! tip "Document Your Experiments"
    ```bash
    # Create an experiment log
    echo "=== My RAG Chatbot Experiments ===" > experiment_log.txt
    echo "Knowledge Domain: [Your chosen domain]" >> experiment_log.txt
    echo "Test Questions:" >> experiment_log.txt
    echo "1. [Your question]" >> experiment_log.txt
    echo "Response: [Chatbot response]" >> experiment_log.txt
    echo "Analysis: [Your observations]" >> experiment_log.txt
    ```

### Restore Original Content

If you want to return to the original OVHcloud content:

```bash
# Restore original content
cp rag-files/content-original.txt rag-files/content.txt
```

**Congratulations!** 🎉 You've now experienced the full power of RAG systems - the ability to create domain-specific AI assistants by simply changing the knowledge base content. This is exactly how production RAG systems work: they combine the reasoning capabilities of large language models with your specific knowledge domain.

## Clean Up

```bash
# Deactivate virtual environment when done
deactivate

# Optional: Save your implementations
mkdir implementations-backup
cp *.py implementations-backup/
cp -r rag-files implementations-backup/

# Optional: Remove demo directory
# rm -rf ~/langchain-chatbot-demo
```


This workbook successfully demonstrates progressive chatbot development using LangChain and OVHcloud AI Endpoints, from basic implementations to advanced RAG systems with streaming and memory capabilities.