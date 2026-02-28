---
title: Deep Persona Factory
emoji: 🎭
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# Deep Persona Factory

Deep Persona Factory is a specialized simulation engine for persona generation and social content testing.

## Features
- **Social Network Engine:** Graph-based modeling and influence propagation.
- **Prediction Engine:** ML and LLM-based engagement scoring.
- **Deep Persona Generation:** Sequential enrichment for high-fidelity character profiles.
- **API Documentation:** Accessible via \`/api-docs\`.
- **Health Check:** Accessible via \`/health\`.

## API Documentation
The application exposes a mandatory \`/api-docs\` endpoint providing Swagger UI for all available endpoints.

## Local Setup
\`\`\`bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
\`\`\`
