# FAB Financial Analysis Agent - Architecture Document

## 1. System Overview

### 1.1 Purpose
The FAB Financial Analysis Agent is a production-ready multi-agent system designed to analyze First Abu Dhabi Bank's (FAB) public financial statements. It provides intelligent financial analysis capabilities including:

- **Multi-hop reasoning**: Questions requiring information from multiple documents
- **Financial calculations**: ROE, ROA, ratios, percentage changes, trend analysis
- **Temporal reasoning**: Cross-quarter and cross-year comparisons
- **Comprehensive synthesis**: Insights combining multiple data points with citations

### 1.2 Design Philosophy
- **Accuracy over Speed**: Financial data must be precise and verifiable
- **Transparency**: All calculations are traced, all sources are cited
- **Modularity**: Each component can be tested and improved independently
- **Production-Ready**: Designed for real banking environments with thousands of queries

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                               │
│                    (CLI / API / Web Interface)                       │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      ORCHESTRATOR (LangGraph)                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │   PLANNER   │─▶│  RETRIEVER  │─▶│ CALCULATOR  │─▶│ SYNTHESIZER │ │
│  │    Agent    │  │    Agent    │  │    Agent    │  │    Agent    │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │
│                                                            │        │
│                                                            ▼        │
│                                                      ┌───────────┐  │
│                                                      │ VERIFIER  │  │
│                                                      │   Agent   │  │
│                                                      └───────────┘  │
└─────────────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   TOOL LAYER    │  │   TOOL LAYER    │  │   TOOL LAYER    │
│  ┌───────────┐  │  │  ┌───────────┐  │  │  ┌───────────┐  │
│  │ Document  │  │  │  │ Financial │  │  │  │ Temporal  │  │
│  │ Retriever │  │  │  │Calculator │  │  │  │ Reasoning │  │
│  └───────────┘  │  │  └───────────┘  │  │  └───────────┘  │
└─────────────────┘  └─────────────────┘  └─────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        VECTOR DATABASE                               │
│                     (ChromaDB + Embeddings)                          │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                  FAB Quarterly Reports Collection                │ │
│  │  • Q1-Q4 2022-2024                                               │ │
│  │  • Earnings Calls, Presentations, Financial Statements          │ │
│  │  • Metadata: quarter, year, report_type, source                 │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Details

### 3.1 Agent Orchestration (LangGraph)

**Framework Choice Justification:**
- LangGraph provides explicit state management and conditional routing
- Better suited for complex, multi-step financial workflows than simple chains
- Supports iterative refinement and error recovery
- Clear visibility into agent execution for debugging and auditing

**Workflow Stages:**

| Stage | Agent | Responsibility |
|-------|-------|----------------|
| 1 | Planner | Analyzes query, identifies data needs, creates execution plan |
| 2 | Retriever | Searches vector DB, extracts financial figures with sources |
| 3 | Calculator | Performs verified calculations with full tracing |
| 4 | Synthesizer | Combines data and calculations into coherent answer |
| 5 | Verifier | Checks for hallucinations, validates citations |

### 3.2 Tool Layer

#### Document Retriever Tool
- **Purpose**: Vector search with metadata filtering
- **Capabilities**:
  - Semantic search using OpenAI embeddings
  - Filter by quarter (Q1-Q4), year, report type
  - Source citation tracking
  - Relevance scoring

#### Financial Calculator Tool
- **Purpose**: Precise financial calculations with tracing
- **Supported Calculations**:
  - `percentage_change`: (new - old) / old × 100
  - `growth_rate`: Year-over-year or quarter-over-quarter
  - `roe`: Net Income / Shareholders' Equity × 100
  - `roa`: Net Income / Total Assets × 100
  - `loan_to_deposit`: Total Loans / Total Deposits × 100
  - `net_interest_margin`: Net Interest Income / Average Earning Assets × 100
  - `cost_to_income`: Operating Expenses / Operating Income × 100
  - `average`, `sum`, `compound_growth` (CAGR)

- **Tracing**: Every calculation includes:
  - Step-by-step formula application
  - Input values with sources
  - Intermediate results
  - Final result with formatting

#### Temporal Reasoning Tool
- **Purpose**: Date and period analysis
- **Capabilities**:
  - Parse period strings (e.g., "Q3 2023", "FY2022")
  - Compare periods (earlier/later/same)
  - Generate period ranges
  - Get YoY/QoQ comparison periods
  - Order periods chronologically

### 3.3 Vector Database (ChromaDB)

**Choice Justification:**
- Lightweight, embedded database suitable for deployment
- Persistent storage without external infrastructure
- Good performance for financial document corpus size
- Native metadata filtering support

**Schema:**
```python
{
    "id": "unique_chunk_id",
    "document": "chunk_content",
    "metadata": {
        "source": "document_filename.pdf",
        "quarter": "Q4",
        "year": 2023,
        "report_type": "earnings_call|presentation|financial_statement",
        "chunk_id": "section_identifier"
    },
    "embedding": [float_vector]
}
```

### 3.4 LLM Selection

**Primary Model: GPT-4o-mini (OpenAI)**
- **Reasoning**: Best balance of cost, speed, and accuracy for financial analysis
- **Cost**: ~$0.15/1M input tokens, ~$0.60/1M output tokens
- **Latency**: Fast enough for interactive use
- **Accuracy**: Strong numerical reasoning capabilities

**Alternative: Claude 3.5 Sonnet (Anthropic)**
- **Reasoning**: Excellent for complex reasoning and longer contexts
- **Cost**: ~$3/1M input tokens, ~$15/1M output tokens
- **Use Case**: When higher accuracy is required, budget permitting

---

## 4. Data Flow

### 4.1 Query Processing Flow

```
User Query
    │
    ▼
┌───────────────────────────────────────┐
│ PLANNER: Parse query, identify needs  │
│ • What data is needed?                │
│ • Which time periods?                 │
│ • What calculations required?         │
└───────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────┐
│ RETRIEVER: Search and extract         │
│ • Vector search with filters          │
│ • Extract numerical values            │
│ • Track source citations              │
└───────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────┐
│ CALCULATOR: Perform calculations      │
│ • Apply formulas with tracing         │
│ • Verify input consistency            │
│ • Format results                      │
└───────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────┐
│ SYNTHESIZER: Generate response        │
│ • Combine data and calculations       │
│ • Add business interpretation         │
│ • Include citations                   │
└───────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────┐
│ VERIFIER: Validate response           │
│ • Check for unsupported claims        │
│ • Verify number traceability          │
│ • Calculate confidence score          │
└───────────────────────────────────────┘
    │
    ▼
Final Response with Citations
```

---

## 5. Hallucination Prevention

### 5.1 Strategies Implemented

1. **Source Citation Requirement**
   - Every factual claim must reference a source document
   - System refuses to answer if sources are insufficient

2. **Calculation Tracing**
   - All calculations show step-by-step work
   - Input values linked to source documents
   - No "magic numbers" in responses

3. **Verification Layer**
   - Post-generation check for unsupported numbers
   - Confidence scoring based on source coverage
   - Uncertainty notes when data is ambiguous

4. **Explicit Refusal**
   - Out-of-scope queries trigger clear refusal messages
   - No fabricated data for missing information
   - Suggestions for alternative queries when helpful

### 5.2 Confidence Scoring

```
Confidence = 1.0 × source_factor × calculation_factor × consistency_factor

source_factor:
  - 1.0: Multiple relevant sources found
  - 0.7: Limited sources (< 3 documents)
  - 0.3: No relevant sources

calculation_factor:
  - 1.0: All calculations verified
  - 0.8: Some unverified numbers in response

consistency_factor:
  - 1.0: No conflicting data
  - 0.7: Minor inconsistencies noted
```

---

## 6. Cost Estimation

### 6.1 Per-Query Cost (GPT-4o-mini)

| Component | Tokens (Avg) | Cost |
|-----------|--------------|------|
| Planner | ~2,000 | $0.0003 |
| Retriever | ~3,000 | $0.0005 |
| Calculator | ~2,000 | $0.0003 |
| Synthesizer | ~5,000 | $0.0008 |
| Verifier | ~1,500 | $0.0002 |
| **Total** | ~13,500 | **$0.002** |

### 6.2 Daily Cost Projection

| Query Volume | Daily Cost | Monthly Cost |
|--------------|------------|--------------|
| 100 queries | $0.20 | $6 |
| 1,000 queries | $2.00 | $60 |
| 10,000 queries | $20.00 | $600 |

### 6.3 Infrastructure Costs

- **ChromaDB**: Self-hosted, no external cost
- **Embeddings** (OpenAI text-embedding-3-small): ~$0.02 per 1M tokens
- **Storage**: Minimal (< 100MB for document corpus)

---

## 7. Scalability Considerations

### 7.1 Current Limitations
- Single-threaded query processing
- In-memory LangGraph state (no persistence)
- Local ChromaDB instance

### 7.2 Scaling Strategies

**Horizontal Scaling:**
- Deploy as stateless API behind load balancer
- Each instance has local ChromaDB replica
- Use Redis for session state if needed

**Database Scaling:**
- Migrate to managed vector DB (Pinecone, Weaviate) for larger corpora
- Implement caching for frequent queries
- Pre-compute common calculations

**Compute Scaling:**
- Use async/await for concurrent LLM calls
- Batch similar queries for efficiency
- Implement query queue for peak load

---

## 8. Known Limitations

### 8.1 Data Limitations
- Only covers FAB documents in the corpus
- Cannot compare with competitor banks
- No real-time market data

### 8.2 Analytical Limitations
- Cannot predict future performance
- Limited to metrics calculable from available data
- May miss nuances requiring domain expertise

### 8.3 Technical Limitations
- Response latency 5-15 seconds for complex queries
- Embedding model may miss financial nuances
- Context window limits for very long analyses

---

## 9. Future Improvements

### 9.1 Short-term (1-3 months)
- [ ] Add streaming responses for better UX
- [ ] Implement query caching
- [ ] Add more financial ratio calculations
- [ ] Improve temporal extraction accuracy

### 9.2 Medium-term (3-6 months)
- [ ] Visualization agent for charts/graphs
- [ ] Self-reflection and error recovery loops
- [ ] Multi-modal support for tables/charts in PDFs
- [ ] LangSmith integration for observability

### 9.3 Long-term (6-12 months)
- [ ] Fine-tuned embedding model for finance
- [ ] Automated report generation
- [ ] Integration with live market data
- [ ] Multi-bank comparative analysis

---

## 10. Technology Stack Summary

| Layer | Technology | Version |
|-------|------------|---------|
| Orchestration | LangGraph | ≥0.2.0 |
| LLM Framework | LangChain | ≥0.2.0 |
| LLM Provider | OpenAI/Anthropic | Latest |
| Vector DB | ChromaDB | ≥0.4.0 |
| Embeddings | OpenAI text-embedding-3-small | Latest |
| Language | Python | 3.11 |
| CLI | Rich | ≥13.0.0 |
| Validation | Pydantic | ≥2.0.0 |

---

## Appendix A: Project Structure

```
fab-financial-agent/
├── main.py                     # Entry point and CLI
├── requirements.txt            # Python dependencies
├── pyproject.toml             # Project configuration
├── .env.example               # Environment template
├── ARCHITECTURE.md            # This document
├── README.md                  # Setup and usage guide
│
├── src/
│   └── fab_financial_agent/
│       ├── __init__.py
│       ├── config.py          # Configuration management
│       ├── data_loader.py     # ChromaDB data setup
│       │
│       ├── agents/
│       │   ├── __init__.py
│       │   ├── orchestrator.py # LangGraph workflow
│       │   └── state.py        # Agent state management
│       │
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── calculator.py   # Financial calculations
│       │   ├── retriever.py    # Document retrieval
│       │   └── temporal.py     # Date/period reasoning
│       │
│       └── evaluation/
│           ├── __init__.py
│           ├── test_cases.py   # 20+ test cases
│           ├── metrics.py      # Evaluation metrics
│           └── runner.py       # Evaluation runner
│
├── chroma_data/               # Vector database storage
└── attached_assets/           # Source data files
```

---

*Document Version: 1.0*
*Last Updated: December 2024*
