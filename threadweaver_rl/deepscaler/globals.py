"""
Global variables for Deepscaler repo.
"""
# Gemini Vertex AI Config (for dataset preprocessing and LLM as ORM).
GCP_PROJECT_ID = None # Fill this in!
GCP_LOCATION = None # Fill this in!
GEMINI_MODEL = "gemini-1.5-pro-002"
OAI_RM_MODEL = "gpt-4o-mini"

# Reward function constants
THOUGHT_DELIMITER_START = "<think>"
THOUGHT_DELIMITER_END = "</think>"

ALTERNATIVE_THOUGHT_DELIMITER_START = "<Think>"
ALTERNATIVE_THOUGHT_DELIMITER_END = "</Think>"

PARALLELIZE_DELIMITER_START = "<Parallel>"
PARALLELIZE_DELIMITER_END = "</Parallel>"
