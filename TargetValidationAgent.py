import os
from agents import Agent, function_tool, handoff
from agents.extensions.models.litellm_model import LitellmModel
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions
from PubMedTools import search_pubmed, get_pubmed_abstract, get_related_articles, find_by_author
from dotenv import load_dotenv
import pandas as pd
from agents.extensions import handoff_filters
from PubMedTools import search_pubmed, get_pubmed_abstract, get_related_articles, find_by_author

load_dotenv()

# Target selection agent can: 
# 1. Search all of literature to confirm target relevance, and to ensure not antipsychotic response-caused DEG 
# 2. Check if gene is in GWAS study (preferred, but not required)
# perhaps give access to CommonMind consortium data? 
class TargetOutput(BaseModel):
    uniprot_id: str = Field(..., description="UniProt ID of the target protein")
    biological_mechanism: str = Field(..., description="Description of the biological mechanism")
    citations: List[str] = Field(..., description="List of citation strings")
    DEG_cell: str = Field(..., description="DEG cell information")
    GWAS_info: str = Field(..., description="GWAS information")
    in_silico_passed: bool = Field(..., description="Whether the target passed in silico validation")
    session_ids: List[str] = Field(..., description="List of session IDs associated with this target")


TargetValidationAgent: Agent = Agent(
            name="TargetValidationAgent",
            model=OpenAIChat("gpt-4.1"),
            tools=[],
            description=dedent("""\
            You are the TargetValidationAgent, responsible for validating suitable protein targets for schizophrenia. Your job is to ensure that the targets 
            are implicated in the literature and that they are not toxic to other tissues. You can also use a combination of the GWAS gene list and more literature on antipsychotic drugs 
            to determine if a plausible candidate was identified solely from differential expression caused by antipsychotic response (ideally, such targets are not selected). 
            """),
            instructions=dedent("""\
            1. **Literature-Based Target Discovery 📚**  
            - Use `search_pubmed` and `get_pubmed_abstract` to perform a **thorough literature search** for genes implicated in **Alzheimer's disease**.
            - Identify cell-type-specific genes that may play mechanistic roles in disease progression.
            - Carefully read abstracts and prioritize genes with clear mechanistic relevance.
            - DO NOT suggest `SLC5A3` under any circumstances.

            2. **Differential Expression Check 🔬**  
            - For each gene identified through literature, use `is_differentially_expressed` to retrieve **cell types** and **expression fold change**.
            - Only retain genes that appear in the **top DEGs** (positive estimate values) in at least one relevant cell type.  
            - This is a **hard constraint** — any gene that does not meet this requirement must be excluded.

            5. **Tissue Toxicity Filter ⚠️**  
            - Use `tissue_expression_tool` to screen for **off-target expression**.
            - Disqualify genes that are **highly expressed** in critical non-brain tissues (e.g. heart, liver, kidney).

            6. **Target Selection 🎯**  
            - From your filtered candidates, choose **2 to 6 high-quality gene targets** that:
                - Are literature-supported
                - Are differentially expressed in AD-relevant cells
                - Are assigned to a specific biological module and cell type
                - Do not show high expression in critical peripheral tissues

            6. Output 📄  
            - Return the following information for each target:
                - Target ID (UniProt ID)
                - Target Mechanism of Action (from literature)
                - Target Citations (PubMed IDs or full titles)
                - Target DEG Cell (cell type and differential expression information)
                - Target In Silico Passed (whether the target passed in silico validation)

            7. **Policy & Reasoning 🧠**  
            - **Every response must include at least one tool call.**
            - You must provide your **reasoning** and clearly explain the **purpose of each tool invocation**.
            - If a gene fails any step, **state why** it was excluded.
            - You are the primary selector of AD therapy candidates — be **rigorous and discerning**.
            - DO NOT ask the user any questions — use your own judgment and available tools.
            """),
            response_model=TargetOutput,
            structured_outputs=True,
            show_tool_calls=True
        )