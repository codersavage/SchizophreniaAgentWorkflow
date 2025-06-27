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
import requests
from utils import convert_pdb_to_pdbqt, compute_vina_box, analyze_protein_ligand_interactions, interaction_table
from Bio.PDB import PDBParser
import numpy as np
import os
import subprocess
import meeko
from agents import Agent, function_tool, handoff
from agents.extensions.models.litellm_model import LitellmModel
from SynthesisAgent import SynthesisAgent
import mygene
import pandas as pd
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions
from agents.extensions import handoff_filters

mg = mygene.MyGeneInfo()


load_dotenv()

# Define model parameters (ensure these match the SOTA model)
drug_shape = 2048
target_shape = 1024
latent_dimension = 1024
patent_shape = 768  # Dimension for patent text embeddings
domain_shape = 768  # Dimension for protein domain embeddings

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model
affinitynet10_model = SimpleCoembedding(
    drug_shape=drug_shape,
    target_shape=target_shape,
    latent_dimension=latent_dimension,
    classify=False,  # For regression tasks like affinity prediction
    patent_shape=patent_shape,
    domain_shape=domain_shape # Ensure this matches the SOTA model if domain embeddings are used
)

# Load pre-trained SOTA weights
affinitynet10_model.load_state_dict(torch.load("AffinityNet10_SOTA.pt", map_location=device))
affinitynet10_model.to(device)
affinitynet10_model.eval() # Set to evaluation mode
all_phase_drugs = pickle.load(open('all_phase_drugs.pkl', 'rb'))
all_prots = pickle.load(open('all_prots_af10.pkl', 'rb'))

def canonicalize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    else:
        return None

class MorganFeaturizer():
    def __init__(
        self,
        shape: int = 2048,
        radius: int = 2,
    ):
        self.shape = shape
        self._radius = radius

    def smiles_to_morgan(self, smile_: str):
        """
        Convert smiles into Morgan Fingerprint.
        :param smile: SMILES string
        :type smile: str
        :return: Morgan fingerprint
        :rtype: np.ndarray
        """
        try:
            smile = canonicalize(smile_)
            mol = Chem.MolFromSmiles(smile)
            features_vec = AllChem.GetHashedMorganFingerprint(mol, self._radius, nBits=self.shape)
            fp_dict = features_vec.GetNonzeroElements()
            features = np.zeros((self.shape,))
            for idx, val in fp_dict.items():
                features[idx] = val
        except Exception as e:
            print(
                f"rdkit not found this smiles for morgan: {smile_} convert to all 0 features"
            )
            print(e)
            features = np.zeros((self.shape,))
        return features

    def _transform(self, smile: str) -> torch.Tensor:
        # feats = torch.from_numpy(self._featurizer(smile)).squeeze().float()
        feats = torch.from_numpy(self.smiles_to_morgan(smile)).squeeze().float()
        if feats.shape[0] != self.shape:
            feats = torch.zeros(self.shape)
        return feats

mol_featurizer = MorganFeaturizer()

def selectivity(target_protein, drug_smiles):
    """
    Compute the selectivity of a drug to a protein using AffinityNet-10 by calculating the affinity.
    
    Args:
        target_protein: The target protein to compute the selectivity for (uniprot id)
        drug_smiles: List of SMILES strings of the drugs to compute the selectivity for
    """
    def batch_selectivity(drug_embeddings, protein_of_interest, nearest_proteins, batch_size=1024):
        """
        Compute selectivity for a batch of drugs
        """
        affinitynet10_model.eval()
        with torch.no_grad():
            # Pre-compute target and domain embeddings for protein of interest
            t = affinitynet10_model.target_projector(protein_of_interest['prott5_embedding'].cuda())
            m = affinitynet10_model.domain_projector(protein_of_interest['domains_pubmed_embedding'].cuda())
            t = t.unsqueeze(0)  # (1, D)
            m = m.unsqueeze(0)  # (1, D)
            
            # Pre-compute embeddings for nearest proteins
            nearest_t = torch.stack([affinitynet10_model.target_projector(p['prott5_embedding'].cuda()) for p in nearest_proteins])
            nearest_m = torch.stack([affinitynet10_model.domain_projector(p['domains_pubmed_embedding'].cuda()) for p in nearest_proteins])
            
            # Process drugs in batches
            all_selectivities = []
            for i in tqdm(range(0, len(drug_embeddings), batch_size)):
                batch_drugs = drug_embeddings[i:i + batch_size].cuda()
                
                # Compute affinity for protein of interest
                d_ = batch_drugs.unsqueeze(1)  # (B, 1, D)
                kv = torch.stack([t.expand(len(batch_drugs), -1), m.expand(len(batch_drugs), -1)], dim=1)
                d_ctx = affinitynet10_model.cross_attn(d_, kv).squeeze(1)
                combined = torch.cat([d_ctx, t.expand(len(batch_drugs), -1), m.expand(len(batch_drugs), -1)], dim=1)
                affinities = affinitynet10_model.affinity_projector(combined).squeeze()
                
                # Compute affinities for nearest proteins
                nearest_affinities = []
                for j in range(0, len(nearest_proteins), 100):  # Process nearest proteins in smaller batches
                    batch_nearest_t = nearest_t[j:j+100]
                    batch_nearest_m = nearest_m[j:j+100]
                    
                    # Expand drug embeddings for this batch
                    d_expanded = d_.expand(-1, len(batch_nearest_t), -1)  # (B, N, D)
                    kv_nearest = torch.stack([batch_nearest_t, batch_nearest_m], dim=1)  # (N, 2, D)
                    kv_nearest = kv_nearest.unsqueeze(0).expand(len(batch_drugs), -1, -1, -1)  # (B, N, 2, D)
                    
                    # Reshape for batch processing
                    d_expanded = d_expanded.reshape(-1, 1, d_expanded.size(-1))  # (B*N, 1, D)
                    kv_nearest = kv_nearest.reshape(-1, 2, kv_nearest.size(-1))  # (B*N, 2, D)
                    
                    d_ctx_nearest = affinitynet10_model.cross_attn(d_expanded, kv_nearest).squeeze(1)
                    combined_nearest = torch.cat([
                        d_ctx_nearest,
                        batch_nearest_t.unsqueeze(0).expand(len(batch_drugs), -1, -1).reshape(-1, batch_nearest_t.size(-1)),
                        batch_nearest_m.unsqueeze(0).expand(len(batch_drugs), -1, -1).reshape(-1, batch_nearest_m.size(-1))
                    ], dim=1)
                    nearest_affinities.append(affinitynet10_model.affinity_projector(combined_nearest).reshape(len(batch_drugs), -1))
                
                nearest_affinities = torch.cat(nearest_affinities, dim=1)
                
                # Compute selectivity scores
                mean_affinity = nearest_affinities.mean(dim=1)
                std_affinity = nearest_affinities.std(dim=1)
                selectivity_scores = (affinities - mean_affinity) / std_affinity
                
                all_selectivities.append(selectivity_scores.cpu())
                
            return torch.cat(all_selectivities)
    

    protein_of_interest = next(p for p in all_prots if p['uniprot_id'] == target_protein)

    nearest_proteins = []

    # Find the nearest 500 proteins to the protein of interest
    all_prots_embeddings = np.array([p['affinitynet10_proj'] for p in all_prots])
    # Use NearestNeighbors to find the nearest proteins
    nbrs = NearestNeighbors(n_neighbors=500, algorithm='ball_tree').fit(all_prots_embeddings)
    distances, indices = nbrs.kneighbors([protein_of_interest['affinitynet10_proj']])  

    indices = indices[0][1:]
    for i in range(len(indices)):
        nearest_proteins.append(all_prots[indices[i]])
    print(f"Found {len(nearest_proteins)} nearest proteins.")

    affinitynet10_model.eval()
    with torch.no_grad():
        drug_embeddings_fp = torch.stack([mol_featurizer._transform(drug) for drug in drug_smiles])
        affinitynet_projs = [affinitynet10_model.drug_projector(fp.cuda()) for fp in drug_embeddings_fp]
        affinitynet_projs = torch.stack(affinitynet_projs)
        selectivity_scores = batch_selectivity(affinitynet_projs, protein_of_interest, nearest_proteins)
    
    return selectivity_scores

@function_tool
def gene_to_uniprot(gene_name: str) -> str:
    x = [gene_name]
    xli = mg.querymany(x, scopes="symbol", fields="uniprot", species="human")
    if xli[0]["uniprot"] is None:
        return f"No Uniprot ID found for gene: {gene_name}"
    return xli[0]["uniprot"]['Swiss-Prot']

@function_tool
def get_top_10_molecules_for(target_protein):
    print('Calling get top 10 drugs for target tool')
    """
    Get the top 10 molecules from phase 1, phase 2, phase 3, and phase 4 for a target protein.

    Args:
        target_protein: The target protein to compute the selectivity for (uniprot id)
    """
    all_smiles = [drug['smiles'] for drug in all_phase_drugs]
    selectivity_scores = selectivity(target_protein, all_smiles)
    sorted_molecules = [all_phase_drugs[i] for i in np.argsort(selectivity_scores)][::-1]
    sorted_selectivity_scores = [selectivity_scores[i].item() for i in np.argsort(selectivity_scores)][::-1]
    top_10_molecules = sorted_molecules[:10]
    for i, molecule in enumerate(top_10_molecules):
        molecule['selectivity_score (Z-score of affinity, higher positive score is better)'] = sorted_selectivity_scores[i]
    return top_10_molecules

def get_pdb_id(uniprot_id: str) -> str:
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.txt"
    response = requests.get(url)
    if response.status_code == 200:
        longest_length = 0
        longest_pdb_id = ""
        for line in response.text.split("\n"):
            if "DR   PDB; " in line:
                pdb_id = line.split("; ")[1]
                prot_length_indices = line.split("=")[-1].replace(".", "")
                begin_index, end_index = prot_length_indices.split("-")
                length = int(end_index) - int(begin_index) + 1
                if length > longest_length:
                    longest_length = length
                    longest_pdb_id = pdb_id
        return longest_pdb_id
    else:
        return "Error: Could not fetch structure"

@function_tool
def qvina_docking(uniprot_id: str, ligand_smiles: str):
    print('Calling QVINA tool')
    pdb_id = get_pdb_id(uniprot_id)
    vina_path = "/home/try_torch/DrugAgent/qvina/bin/qvina-w"
    # Download the pdb file if it doesn't exist in pdb_files
    if not os.path.exists(f"pdb_files/{pdb_id}.pdb"):
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        response = requests.get(url)
        with open(f"pdb_files/{pdb_id}.pdb", "w") as f:
            f.write(response.text)
    # Convert pdb to pdbqt
    if not os.path.exists(f"pdb_files/{pdb_id}.pdbqt"):
        convert_pdb_to_pdbqt(f"pdb_files/{pdb_id}.pdb", f"pdb_files/{pdb_id}.pdbqt")
    center, sizes = compute_vina_box(f"pdb_files/{pdb_id}.pdb")
    lig = Chem.MolFromSmiles(ligand_smiles)
    protonated_lig = Chem.AddHs(lig)
    Chem.AllChem.EmbedMolecule(protonated_lig)
    meeko_prep = meeko.MoleculePreparation()
    meeko_prep.prepare(protonated_lig)
    meeko_prep.write_pdbqt_file(f"ligand.pdbqt")
    cmd = f"{vina_path} --receptor pdb_files/{pdb_id}.pdbqt --ligand ligand.pdbqt --center_x {center[0]} --center_y {center[1]} --center_z {center[2]} --size_x {sizes[0]} --size_y {sizes[1]} --size_z {sizes[2]} --out output.pdbqt --log output.log"
    os.system(cmd)
    interactions = analyze_protein_ligand_interactions(f"pdb_files/{pdb_id}.pdbqt", "output.pdbqt", 5)
    interactions_table = interaction_table(interactions)
    # Get qvina results (just the table) all the lines after and including "mode |"
    qvina_results_table = ""
    start = False   
    with open("output.log", "r") as f:
        for line in f:
            if "mode |" in line:
                start = True
            if start:
                qvina_results_table += line
    return "Docking Results:\n" + qvina_results_table + "\n\nProtein-Ligand Interactions:\n" + interactions_table

DrugSelectionAgent = Agent(
    name="Drug Selection agent",
    instructions=prompt_with_handoff_instructions('''
You are the Drug Selection Agent in a multi-agent drug development workflow. You will be provided with a list of at least 2 validated target proteins, each associated with a gene selected for combination therapy in a human disease context.

Your responsibilities are as follows:

1. Identify candidate small molecule drugs:
   - For each target gene, retrieve its UniProt ID using the gene_to_uniprot tool.
   - Then, use get_top_10_molecules_for with the UniProt ID to retrieve a list of small molecule drugs that have passed Phase I clinical trials.
   - You must identify one suitable drug per gene for inclusion in the final combination therapy. The drug you select does not need to be the first in the list — any molecule in the top 10 is acceptable.

2. Evaluate docking affinity and selectivity:
   - Use qvina_docking to dock each drug candidate to the corresponding protein.
   - Prioritize molecules that:
     - Have a docking score (affinity) greater than 7 (indicating strong binding).
     - Have a positive selectivity score. If the selectivity score is not positive, the molecule is NOT suitable.
   - You should try multiple candidates as needed to find the most suitable drug per target.

3. Filter and finalize:
   - Select one final drug per target gene that satisfies all the above criteria.
   - Ensure that the selected molecules together form a feasible combination therapy (i.e., no contraindications inferred from poor docking or low selectivity).
   - If no suitable molecule is found for a given target, send a message back to the TargetValidationAgent so it can regenerate a new set of targets. Do not proceed with an unsuitable molecule.

4. Handoff:
   - Once all selected molecules meet the criteria (affinity > 7 and positive selectivity), hand off the result to the SynthesisAgent for chemical report generation.

Tools available to you:
- gene_to_uniprot: convert gene names to UniProt IDs
- get_top_10_molecules_for: retrieve top Phase I drugs for a given UniProt ID
- qvina_docking: perform docking analysis and retrieve affinity and selectivity scores

IMPORTANT: Every response you make must include at least one tool call. DO NOT ASK THE USER QUESTIONS, use your discretion. 
Good luck, and thank you for your service.
'''),
    model=LitellmModel(model="gemini/gemini-2.5-pro-preview-05-06", api_key=os.getenv('GEMINI_KEY')),
    tools=[gene_to_uniprot, get_top_10_molecules_for, qvina_docking],
    handoffs=[]  # Initialize with empty handoffs
)

# Import and set handoffs after DrugSelectionAgent is defined
from TargetValidationAgent import TargetValidationAgent
from SynthesisAgent import SynthesisAgent

DrugSelectionAgent.handoffs = [TargetValidationAgent, SynthesisAgent]
