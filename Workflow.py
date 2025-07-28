from CorpusTools import search_corpus, get_paper, get_all_papers
from PubMedTools import search_pubmed, get_pubmed_abstract, get_related_articles, find_by_author

from MainExpressionAgent import MainExpressionAgent
from TargetValidationAgent import TargetValidationAgent

class TargetSelectionWorkflow(Workflow):
    # Do workflow
    # Then save trace to database along with ensuring the target returned also has the current session id
    def __init__(self, mcp_tools=None):
        super().__init__()
        self.mcp_tools = mcp_tools

        self.main_expression_agent: Agent = MainExpressionAgent
        self.target_validation_agent: Agent = TargetValidationAgent

    async def arun(
        self,
        uniprot_id: str,
        biological_mechanism: str, 
        citations: str,
        DEG_cell: str,
        other_vital_tissues: str,
        gwas_info: str,
        in_silico_passed: bool,
    ) -> AsyncIterator[RunResponse]:
        
        # Initialize MCP tools if not already done
        if not self.mcp_tools:
            async with MCPTools(url=SERVER_URL, transport="streamable-http") as mcp_tools:
                self.mcp_tools = mcp_tools

        # 1. Select target candidates
        print("Step 1: MainExpressionAgent identifying potential targets")
        self.mcp_tools.include_tools = ["tissue_expression_tool", "search_corpus", "get_paper", "get_all_papers"]
        async for response in await self.main_expression_agent.arun(dedent(f"""Target ID: {uniprot_id}
        Target Mechanism of Action: {biological_mechanism}
        Target Citations: {citations}
        Target DEG Cell: {DEG_cell}
        Target Other Vital Tissues: {other_vital_tissues}"""), stream=True):
            yield response
        
        # 2. Validate candidate targets 
        print("Step 2: TargetValidationAgent validating targets")
        self.mcp_tools.include_tools = ["search_pubmed", "get_pubmed_abstract", "get_related_articles", "find_by_author"]
        async for response in await self.target_validation_agent.arun(dedent(f"""Target ID: {uniprot_id}
        Target Mechanism of Action: {biological_mechanism}
        Target Citations: {citations}
        Target DEG Cell: {DEG_cell}
        Target GWAS Info: {gwas_info}
        Target In Silico Passed: {in_silico_passed}"""), stream=True):
            yield response
        
        # Yield a completion response
        yield RunResponse(run_id=self.run_id, content=RunEvent.workflow_completed)
        return