# Manzilix-Education-Counselling
# Manzilix — Digital Platform for Education Counselling

Manzilix is a digital education counselling platform that helps students get guidance on courses, career paths, and academic decisions through an AI-powered assistant.

## Features

- 🤖 AI-powered counselling assistant (chat-based Q&A, recommendations, summarization)
- 🎓 Personalized course and career guidance
- 📄 Document and content summarization (brochures, syllabi, etc.)
- ⚡ Fast, responsive user experience

## Tech Stack

- **LLM:** Evaluating NVIDIA Nemotron 3 Ultra (via NIM API) and Claude Haiku 4.5 / Sonnet 4.6
- **Frontend:** _TBD_
- **Backend:** _TBD_
- **Database:** _TBD_

## Project Status

🚧 In development

## Getting Started

```bash
git clone https://github.com/<your-username>/Manzilix-Education-Counselling.git
cd Manzilix-Education-Counselling
```

### LLM Module Setup

The `llm.py` module wraps the chat model used by the assistant (currently NVIDIA Nemotron via the NIM API).

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in your real API key
export NVIDIA_API_KEY="your-real-key"   # PowerShell: setx NVIDIA_API_KEY "your-real-key"
python test_llm.py
```

> ⚠️ Never commit real API keys. `.env` is gitignored — only `.env.example` should be tracked.

_Further setup instructions coming soon._

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
