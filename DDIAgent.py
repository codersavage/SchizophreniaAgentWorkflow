import pickle
import numpy as np
from conplex_architectures_affinitynet10 import SimpleCoembedding
import torch
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect   
from sklearn.neighbors import NearestNeighbors
from rdkit import Chem
from tqdm import tqdm
import os
from dotenv import load_dotenv 
import numpy as np
import os
import subprocess
import meeko
from agents import Agent, function_tool
from agents.extensions.models.litellm_model import LitellmModel
from MainExpressionAgent import MainExpressionAgent
from SynthesisAgent import SynthesisAgent
from DrugSelectionAgent import DrugSelectionAgent
import pandas as pd
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions

load_dotenv()

@function_tool
def get_ddi_toxicity(drug1, drug2):
    # placeholder 
    print('DDI tool called')
    return f'{drug1} and {drug2} are safe together'

DDIAgent = Agent(
    name="Library agent",
    instructions=prompt_with_handoff_instructions('''You will be provided with a set of at least 2 drugs along with a corresponding target protein that 
    each drug is intended to target. You will assess the toxicity of administering those 
    drugs simultaneously using the get_ddi_toxicity tool. If that combination of drugs looks safe for human consumption, 
    you will hand that set of drugs off to the SynthesisAgent for generation of a final report. Otherwise, 
    if you believe it is an issue in the way the targets were chosen, hand 
    it back to the MainExpressionAgent. If you believe it is an issue with the drugs themselves, hand it back to the 
    DrugSelectionAgent. DO NOT ASK THE USER QUESTIONS, use your discretion. Your job is critical. Make sure to pass all information to the SynthesisAgent. 
    IMPORTANT: Every response you make must include at least one tool call.'''),
    model=LitellmModel(model="gemini/gemini-2.5-pro-preview-05-06", api_key=os.getenv('GEMINI_KEY')),
    tools=[get_ddi_toxicity]
    handoffs = [MainExpressionAgent, DrugSelectionAgent, SynthesisAgent]  
) 

