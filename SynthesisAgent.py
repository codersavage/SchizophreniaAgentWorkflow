from agents import Agent, function_tool
from agents.extensions.models.litellm_model import LitellmModel
import os
from dotenv import load_dotenv
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions
load_dotenv()

@function_tool
def make_synthesis_report(target_proteins: list[str], drugs: list[str], trial_phases: list[str], vina_scores: list[float], mechanisms: list[str], modules: list[int], cell_types: list[str], overall_mechanism: str):
    """
    Make a synthesis report from the information provided.
    Args:
        target_proteins: list of target proteins (uniprot ids)
        drugs: list of drugs (smiles)
        trial_phases: list of clinical trial phases of compounds (Phase 1, Phase 2, Phase 3, etc.)
        vina_scores: list of vina docking scores in kcal/mol
        mechanisms: list of mechanisms of action
        overall_mechanism: overall mechanism of action
    """
    print('Make synthesis report tool called')
    final_report = "## Synthesis Report\n"
    # Make a md table with the following columns: target protein (uniprot id), drug (smiles), Drug Clinical Trial Phase, Vina score (kcal/mol), mechanism of action
    final_report += "| Target Protein | Drug | Drug Clinical Trial Phase | Vina score (kcal/mol) | Mechanism of Action | Module Number | Cell Type |\n"
    final_report += "|----------------|------|---------------------------|-----------------------|---------------------|---------------|-----------|\n"
    for target_protein, drug, phase, score, mechanism, module_, cell_type in zip(target_proteins, drugs, trial_phases, vina_scores, mechanisms, modules, cell_types):
        final_report += f"| {target_protein} | {drug} | {phase} | {score:.1f} | {mechanism} | {module_} | {cell_type} \n"
    final_report += f"\n## Overall Mechanism of Action\n{overall_mechanism}"
    open('synthesis_report.md', 'w').write(final_report)
    return "Synthesis report has been written to synthesis_report.md"

SynthesisAgent = Agent(
    name="Synthesis agent",
    instructions=prompt_with_handoff_instructions( '''You will compile the information from all the agents. First, have a table
    with the following columns: target protein (uniprot id), drug (smiles), mechanism of action (with citations and evidence). 
    Also, provide a summary of the overall mechanism of action of the combination of drugs used with citations and evidence.
    Add your results to the synthesis_report using the make_synthesis_report tool. Once you have done this, you may complete. 
    IMPORTANT: Every response you make must include at least one tool call. 
    DO NOT ASK THE USER QUESTIONS, use your discretion. 
    Good luck, and thank you for your service.'''),
    model=LitellmModel(model="gemini/gemini-2.5-pro-preview-05-06", api_key=os.getenv('GEMINI_KEY')),
    tools=[make_synthesis_report] 
) 