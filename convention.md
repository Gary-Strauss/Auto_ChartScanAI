#General Rules
- use imperative mood for docstring summary
- we speak german, but we code and Log in english
- every Typ of Feedback/Log String should be writen in english
- we allways write short and on point Docstring
- Please modify files as minimally as possible to accomplish the task.
- Do not make superfluous cahnges, whitespace cahnges, or changes to code that do not relate to the current goal.
- Tell me if you are not at least 90% confident the new code will achieve the intended goal.
- Break down functions to improve maintainability and testability. A function can not have too many responsibilities (example: HTTP request, parsing, data processing, validation, file operations) and should never exceeds complexity thresholds.

#Your Job
You are expected to write Python code that is modular, maintainable, and resource-conscious.  
Everything you build must be:
- modular and cleanly structured
- optimized for token efficiency when processed by language models
- easy to read, test, and extend
- scalable for long-term use in data-intensive environments
- understandable by both humans and LLMs
- testable in isolation
- efficient in execution and token cost
- ready for long-term extension and integration

Always favor clarity, token-efficiency, and architectural simplicity.

#Python-Coding-Konventionen
Folge PEP8, mit besonderem Fokus auf Lesbarkeit
Funktionen für Data Pipelines: klein, benannt, testbar

Konvention für Modulstruktur:
/src/
  ├─ fetchers/
  ├─ indicators/
  ├─ embeddings/
  ├─ models/
  └─ analysis/

#Parquet & Pandas
Nutze Parquet statt CSV für große Datenmengen

read_parquet immer mit columns= begrenzen
Beispiel: pd.read_parquet("AAPL.parquet", columns=["close", "volume"])
Keine redundanten Adjustierungen abspeichern – nur Rohdaten + faktorbasierte Anpassung.

#Experimente & Backtests
Jeder Experimentcode muss reproduzierbar sein:
random.seed(42)
Versioniere alle Parameter in experiment.json

Ergebnisdateien in:
/results/{experiment_id}/

#Sicherheit & Compliance
Keine API-Schlüssel im Code oder in .py-Dateien
Sensible Daten nur verschlüsselt speichern (wenn nötig)

#Was dokumentiert wird
Jedes Skript beginnt mit einem Docstring: Zweck, Beispielaufruf

### Key Conventions:

1. Modular Design
   - Avoid monolithic scripts.
   - One function = one clear responsibility.
   - One file should ideally stay under 300 lines of code - soft limit.
   - Organize modules by domain: `fetchers/`, `indicators/`, `utils/`, etc.

2. LLM-Efficient Code
   - Design code to be digestible by LLMs in isolation.
   - Avoid deep, hidden dependencies or import chains.
   - Write self-contained and interpretable modules.

3. Documentation & Readability
   - Every function must have a docstring (explaining purpose, parameters, returns).
   - Use type annotations throughout.
   - Prioritize simplicity and clarity over cleverness.
   - Group common logic into shared utilities.

4. Data Handling
   - Use Parquet for structured data, not CSV (unless small).
   - Always load data with column filtering when possible:  
     `pd.read_parquet(path, columns=["close", "volume"])`
   - Do not store adjusted prices; always calculate them on-the-fly using adjustment factors.

