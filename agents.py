"""
agents.py
Orquestração linear e simples entre dois provedores de LLM:
1. Groq (Llama 3)      -> triagem rápida de sentimento
2. Azure AI Foundry     -> diagnóstico/sugestão em texto puro

Design propositalmente simples: sem JSON estruturado entre etapas,
sem função calling, sem retries complexos. O objetivo é robustez,
não sofisticação.
"""

import os
from groq import Groq
# from openai import AzureOpenAI
from openai import OpenAI


class SimpleOrchestrator:
    def __init__(self):
        # --- Cliente Groq ---
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        # --- Cliente Azure AI Foundry (via SDK OpenAI compatível) ---
        self.azure_client = OpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            base_url=f"{os.getenv('AZURE_OPENAI_ENDPOINT')}/openai/v1",
            # api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
        )
        self.azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5")

    # ------------------------------------------------------------------
    # ETAPA 1 — Groq: triagem rápida de sentimento
    # ------------------------------------------------------------------
    def _analyze_sentiment_groq(self, text: str) -> str:
        """
        Retorna uma string simples: POSITIVO, NEGATIVO ou NEUTRO.
        Evitamos pedir JSON ao modelo — pedimos uma única palavra, 
        e ainda assim validamos o resultado antes de aceitar.
        """
        try:
            response = self.groq_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {
                        "role": "system",
                        "content": (
                         "Você é um classificador de sentimentos de alta precisão. "
                        "Analise o texto e responda APENAS com uma destas três palavras, sem pontuação: "
                        "POSITIVO, NEGATIVO ou NEUTRO.\n\n"
                        "Exemplos de referência:\n"
                        "- NEGATIVO: termos como 'odiei', 'péssimo produto', 'preciso devolver'.\n"
                        "- POSITIVO: termos como 'amei', 'excelente', 'produto poderoso', 'vou pedir mais'."
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                temperature=0,
                max_tokens=10,
            )

            raw = response.choices[0].message.content.strip().upper()

            # Validação defensiva: se o modelo "enfeitar" a resposta,
            # ainda conseguimos extrair a categoria correta.
            for tag in ["POSITIVO", "NEGATIVO", "NEUTRO"]:
                if tag in raw:
                    return tag

            return "NEUTRO"  # fallback seguro

        except Exception as e:
            return f"ERRO_GROQ: {str(e)}"

    # ------------------------------------------------------------------
    # ETAPA 2 — Azure AI Foundry: diagnóstico em texto puro
    # ------------------------------------------------------------------
    def _generate_diagnosis_azure(self, text: str, sentiment: str) -> str:
        """
        Recebe o texto original + o sentimento do Groq e pede um
        diagnóstico curto. Pedimos explicitamente texto puro para
        evitar quebra por formatação Markdown/JSON inesperada.
        """
        try:
            prompt = (
                f'Texto original do usuário: "{text}"\n'
                f"Sentimento pré-classificado: {sentiment}\n\n"
                "Com base nisso, escreva um diagnóstico curto (2 a 3 frases) "
                "e uma sugestão prática de resposta ou ação. "
                "Responda em texto corrido, sem listas, sem markdown, sem JSON."
            )

            response = self.azure_client.chat.completions.create(
                model=self.azure_deployment,  # nome do deployment no Azure
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Você é um analista que fornece diagnósticos objetivos "
                            "e sugestões práticas em texto puro."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                # temperature=0.4,
                max_completion_tokens=10000,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            # import traceback
            # traceback()
            return f"ERRO_AZURE: {str(e)}"

    # ------------------------------------------------------------------
    # Orquestração linear (ponto de entrada)
    # ------------------------------------------------------------------
    def run(self, text: str) -> dict:
        if not text or not text.strip():
            return {"error": "Texto vazio."}

        # Etapa 1: Groq
        sentiment = self._analyze_sentiment_groq(text)

        # Etapa 2: Azure (recebe o resultado da etapa 1)
        diagnosis = self._generate_diagnosis_azure(text, sentiment)

        return {
            "input_text": text,
            "groq_sentiment": sentiment,
            "azure_diagnosis": diagnosis,
        }