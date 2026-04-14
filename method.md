


                                  ┌──────────────────────────────────┐
                    ┌────────────>│  Optimizer-->Strategy (v0→v1→v2) │<──────────────┬
                    │             └──┬──────────────────────────┬────┘               │
                    │            building                     retrieval              │
                    │            strategy                     strategy               │
                    │                ↓                          ↓                    │
                    │         ┌───────┐    ┌─────────┐       ┌──────────┐            │
  Streaming ──→ Adaptive ────→│Builder│──→ │ Memory  │ ─────→│ Answerer │──> Answer  │
  Text          Buffer        └───────┘    │  Graph  │       └──────────┘            │
                    │                      └─────────┘            ↑                  │
                    │                        ↓      ↓             │                  │
                    │                    generate  retrieve    answers               │
                    │                    QA pairs  evidence       │                  │
                    │                  ┌────────────────┐   ┌────────────┐           │
                    └──────────────────│   Questioner   │───│ Optimizer  │───────────└
                                       └────────────────┘   │ evaluate   │
                                                            │ reflect    │
                                                            │ rollback   │
                                                            └─────┬──────┘
                                                                  │ updates
                                                                  ↑ Strategy
                                                     ┌─────────────────────┐
                                                     │Multi-Level Interven.│
                                                     │ L1→L2→L3→L4         │
                                                     └─────────────────────┘

