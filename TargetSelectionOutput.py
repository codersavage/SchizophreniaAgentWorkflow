import pandas as pd
import os
from textwrap import dedent
from dotenv import load_dotenv
from CorpusTools import search_corpus, get_paper, get_all_papers
load_dotenv()
# Main expression agent can:
# 1. search literature 
# 2. read combined schizophrenia sc rna data for diff expression
# 3. tissue expression tool to evaluate expression in other vital tissues, a measure of toxicity
class TargetSelection_Output(BaseModel):
    uniprot_id: str = Field(..., description="UniProt ID of the target protein")
    biological_mechanism: str = Field(..., description="Description of the biological mechanism")
    citations: List[str] = Field(..., description="List of citation strings")
    DEG_cell: str = Field(..., description="DEG cell information")
    other_vital_tissues: str = Field(..., description="Other vital tissues that the target is expressed in")

TargetSelectionAgent: Agent = Agent(
            name="MainExpressionAgent",
            model=OpenAIChat("gpt-4.1"),
            tools=[],
            description=dedent("""\
            You are the MainExpressionAgent, responsible for identifying suitable protein targets for schizophrenia. Your job is to ensure that the targets 
            are differentially expressed in cells affected by the disease and that those targets are implicated in the literature.
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
            response_model=MainExpOutput,
            structured_outputs=True,
            show_tool_calls=True
        )

# @function_tool
# def get_alzheimers_cell_types():
#     """
#     Get the cell types of interest for AD.
#     """
#     print('Get AD cell types tool called')
#     # As of now, we're only interested in the following cell types:
#     return ['Mic', 'Ast', 'Oli', 'Inh PVALB HTR4', 'Exc']

# @function_tool
# def is_differentially_expressed(gene_name: str):
#     """
#     Check if a gene is differentially expressed in any of the cell types of interest for AD.

#     Args:
#         gene_name: the gene of interest
#     Returns:
#         list of cell types in which the gene is differentially expressed, None if it is not differentially expressed
#     """
#     print('Get cells in which gene is expressed tool called')
#     df = pd.read_csv('filtered_combined.csv')
#     df = df[df['gene'] == gene_name]
#     df = df[df['Estimate'] > 0]
#     df = df.sort_values(by='Estimate', ascending=False)
#     df['cell_type'] = [file_.split('_')[0] for file_ in df['source_file'].values]
#     df = df.drop(columns=['Unnamed: 0', 'source_file'])
#     return df.to_dict(orient="records")

# @function_tool
# def tissue_expression_tool(gene_name: str) -> str:
#     """
#     Used for toxicity assessment. Get the expression data across tissues for a given gene.

#     Args:
#         gene_name: the gene of interest
#     Returns:
#         list of tissues and their expression levels
#     """
#     print('Tissue expression tool called')
#     df = pd.read_csv('rna_tissue_consensus.tsv', sep='\t')
#     df = df[df['Gene name'] == gene_name]
#     if df.empty:
#         return f"No expression data found for gene: {gene_name}"
#     expression_data = df[["Tissue", "nTPM"]].to_dict(orient="records")
#     return expression_data

# @function_tool
# def add_checkpoint_report(target_genes: list[str], mechanisms: list[str], evidence: list[str], citations: list[str]):
#     """
#     Add a checkpoint report to the checkpoint_report.md file.

#     Args:
#         target_genes: list of target genes
#         mechanisms: list of mechanisms corresponding to the target genes
#         evidence: list of evidence corresponding to the target genes
#         citations: list of citations corresponding to the target genes
#     """
#     print('Add checkpoint report tool called')
#     # Call it checkpoint_{r}.md where r is the next checkpoint number. Find r by looking at the files in the folder.
#     r = 0
#     while os.path.exists(f'checkpoints/checkpoint_{r}.md'):
#         r += 1
#     # Turn into a table with the following columns: target gene, mechanism, evidence, citation
#     table = "| Target Gene | Mechanism | Evidence | Citation |\n"
#     table += "|-------------|-----------|----------|----------|\n"
#     for target_gene, mechanism, evidence, citation in zip(target_genes, mechanisms, evidence, citations):
#         table += f"| {target_gene} | {mechanism} | {evidence} | {citation} |\n"
#     with open(f'checkpoints/checkpoint_{r}.md', 'w') as f:
#         f.write(table)


# MainExpressionAgent = Agent(
#     name="Main Expression agent",
#     instructions=prompt_with_handoff_instructions('''Your job is to identify candidate target genes for Alzheimer's disease (AD) therapy by first conducting a thorough literature search (using `search_corpus` and `get_paper`) on cell-type-specific mechanisms implicated in AD. From this, gather a shortlist of potential targets (genes of interest).

# Then, for each shortlisted gene, use the is_differentially_expressed function to get the cell_type and fold change, if applicable, in which the gene is differentially expressed. Only retain genes that appear in the top DEGs of at least one relevant cell type — this is a hard constraint. YOU DO NOT HAVE ACCESS TO `tissue_expression_tool`. YOU DO NOT HAVE ACCESS TO `get_module_number_and_cell_type`. 

# You must choose **2 to 4 target genes** (minimum 2, maximum 4) that are likely useful for *inhibition*-based COMBINATION therapies (so both targets will be inhibited) and supported by **both** literature evidence and DEG ranking. Your output must include:

# 1. **Checkpoint report** with the following columns:
#     - `target gene`
#     - `mechanism`
#     - `evidence`
#     - `citation`
# 2. A **handoff to the TargetValidationAgent** with your final set of targets.

# IMPORTANT CONSTRAINTS:
# - The genes must be *initially* derived from literature review, *then* validated by DEG rankings (not the other way around).
# - DO NOT SUGGEST `SLC5A3` under any circumstances.
# - Every single response you make MUST include at least one tool call.
# - Be thorough: read abstracts in detail, evaluate multiple candidate genes, and don’t just select the first ones you find.
# - The literature corpus is already Alzheimer’s-focused, so you do NOT need to include "Alzheimer's" in your queries.
# - When a set of genes are deemed unsuccessful by the TargetValidationAgent and are handed back to you for refinement, DO NOT propose a set of genes you have already proposed in the past. 
# - When using tools, make sure to provide your reasoning/thought process while doing so.

                                                  
# You are the first and most critical agent in the workflow. Your decisions will determine the downstream synthesis and therapeutic design. Be rigorous and discerning.
# DO NOT ASK THE USER QUESTIONS, use your discretion. Good luck, and thank you for your service.
# '''),
#     model=LitellmModel(model="gemini/gemini-2.5-pro-preview-05-06", api_key=os.getenv('GEMINI_KEY')),
#     tools=[
#         get_alzheimers_cell_types,
#         is_differentially_expressed,
#         search_corpus,
#         get_paper,
#         get_all_papers,
#         add_checkpoint_report
#     ],
#     handoffs=[]  # Will be set dynamically during runtime
# )
# # Import and set handoffs after MainExpressionAgent is defined
# from TargetValidationAgent import TargetValidationAgent
# MainExpressionAgent.handoffs = [TargetValidationAgent]
