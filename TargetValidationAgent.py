import os
from agents import Agent, function_tool, handoff
from agents.extensions.models.litellm_model import LitellmModel
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions
from PubMedTools import search_pubmed, get_pubmed_abstract, get_related_articles, find_by_author
from dotenv import load_dotenv
import pandas as pd
from agents.extensions import handoff_filters
from CorpusTools import search_corpus, get_paper, get_all_papers

load_dotenv()

@function_tool
def get_module_number_and_cell_type(gene_name: str) -> dict:
    """
    Used for pathway targeting assessment. Retrieves the module number and cell type of a gene (does not 
    necessarily imply the gene is implicated in Alzheimer's disease). If two genes are in the same module, 
    they are likely to be co-regulated. 

    Args:
        gene_name: the gene of interest
    Returns:
        the module and cell type of which gene is critical to 
    """
    df = pd.read_csv('filtered_combined_modules.csv')
    df = df[df['gene'] == gene_name]
    df = df[df['set'] != 'All']
    df = df.rename(columns={'set' : 'cell_type'})
    if df.empty:
        return f"{gene_name} is not a core gene."
    module_data = df[["module", "cell_type"]].to_dict(orient="records")
    return module_data#, f'The gene {gene_name} is a core gene in module {df.loc[0, 'module']} in cell type {df.loc[0, 'cell_type']}'

@function_tool
def tissue_expression_tool(gene_name: str) -> str:
    """
    Used for toxicity assessment. Get the expression data across tissues for a given gene.

    Args:
        gene_name: the gene of interest
    Returns:
        list of tissues and their expression levels
    """
    df = pd.read_csv('rna_tissue_consensus.tsv', sep='\t')
    df = df[df['Gene name'] == gene_name]
    if df.empty:
        return f"No expression data found for gene: {gene_name}"
    expression_data = df[["Tissue", "nTPM"]].to_dict(orient="records")
    return expression_data

@function_tool
def add_checkpoint_report(target_genes: list[str], mechanisms: list[str], evidence: list[str], citations: list[str], safety_profile: list[str], biomarkers: list[str]):
    """
    Add a checkpoint report to the checkpoint_report.md file.

    Args:
        target_genes: list of target genes
        mechanisms: list of mechanisms corresponding to the target genes
        evidence: list of evidence corresponding to the target genes
        citations: list of citations corresponding to the target genes
        safety_profile: list of safety profiles corresponding to the target genes
        biomarkers: list of biomarkers corresponding to the target genes
    """
    print('Add checkpoint report tool called')
    # Call it checkpoint_{r}.md where r is the next checkpoint number. Find r by looking at the files in the folder.
    r = 0
    while os.path.exists(f'checkpoints/checkpoint_{r}.md'):
        r += 1
    # Turn into a table with the following columns: target gene, mechanism, evidence, citation
    table = "| Final Target Gene | Mechanism | Evidence | Citation | Safety Profile | Biomarkers | Module and Cell Type |\n"
    table += "|-------------|-----------|----------|----------|---------------|------------|-------------|\n"
    for target_gene, mechanism, evidence, citation, safety_profile, biomarkers in zip(target_genes, mechanisms, evidence, citations, safety_profile, biomarkers):
        table += f"| {target_gene} | {mechanism} | {evidence} | {citation} | {safety_profile} | {biomarkers} |\n"
    with open(f'checkpoints/checkpoint_{r}.md', 'w') as f:
        f.write(table)

TargetValidationAgent = Agent(
    name="Target Validation agent",
    instructions=prompt_with_handoff_instructions('''
You are the Target Validation Agent in a multi-agent workflow focused on designing combination therapies for neurodegenerative disease. Your task is to validate a set of gene targets proposed by another agent.

Your responsibilities are as follows:

1. Validate target suitability:
   - Validate that some combination of gene targets provided can be inhibited *together* in a combination therapy setting.
   - Ensure that selected targets are not highly expressed in critical, non-brain tissues such as the heart, liver, and kidney.
   - Ensure that no two selected targets belong to the same biological module or pathway. Targets in the same module are considered redundant and should not be selected.

2. Disqualify low-quality targets:
   - If a gene does not return both a module number and cell type using get_module_number_and_cell_type, discard it — it is likely not implicated in Alzheimer's disease.
   - If a gene has high expression in vital organs according to tissue_expression_tool, it should not be selected.

3. Support experimental follow-up:
   - For each selected target, identify one or more biomarkers that can be experimentally measured to confirm target inhibition.
   - Support all biomarker recommendations with literature citations using tools such as search_pubmed, get_pubmed_abstract, and get_related_articles.

4. Report your findings:
   - If none of the proposed targets meet the criteria, send feedback to the MainExpressionAgent so it can regenerate a new set of targets.
   - Once valid targets are selected, use add_checkpoint_report to log your decisions to checkpoint_report.md. Each row of the report should include:
     - Gene name
     - Module number
     - Cell type
     - Tissue expression summary
     - Proposed biomarker(s)
     - Supporting literature references

5. Handoff:
   - After successful validation, hand off the selected targets to the DrugSelectionAgent for downstream processing.

Tools available to you:
- search_pubmed: search literature for genes or biomarkers
- get_pubmed_abstract: retrieve abstracts of PubMed articles
- get_related_articles: find articles related to a topic or gene
- find_by_author: find all articles by a specific author
- tissue_expression_tool: obtain expression profiles of genes across tissues
- get_module_number_and_cell_type: retrieve module number and associated cell type for a gene
- add_checkpoint_report: add your findings to checkpoint_report.md

IMPORTANT: Every response you make must include at least one tool call. If you feel that the targets are not valid, you must handoff to the MainExpressionAgent to regenerate a new set of targets.
DO NOT ASK THE USER QUESTIONS, use your discretion. Good luck, and thank you for your service.
'''),
    model=LitellmModel(model="gemini/gemini-2.5-pro-preview-05-06", api_key=os.getenv('GEMINI_KEY')),
    tools=[tissue_expression_tool, search_pubmed, get_pubmed_abstract, get_related_articles, find_by_author, add_checkpoint_report, get_module_number_and_cell_type],
    handoffs = []
    # table of genes + evidence types and their scores so that human can see what genes 
    # and why they were chosen 
)

# Import and set handoffs after TargetValidationAgent is defined
from MainExpressionAgent import MainExpressionAgent
from DrugSelectionAgent import DrugSelectionAgent


TargetValidationAgent.handoffs.append(MainExpressionAgent) 
TargetValidationAgent.handoffs.append(DrugSelectionAgent)