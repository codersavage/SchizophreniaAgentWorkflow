class TargetOutput(BaseModel):
    uniprot_id: str = Field(..., description="UniProt ID of the target protein")
    biological_mechanism: str = Field(..., description="Description of the biological mechanism")
    citations: List[str] = Field(..., description="List of citation strings")
    DEG_cell: str = Field(..., description="DEG cell information")
    module_number: int = Field(..., description="Module number for categorization")
    in_silico_passed: bool = Field(..., description="Whether the target passed in silico validation")
    session_ids: List[str] = Field(..., description="List of session IDs associated with this target")

class TargetSelectionWorkflow(Workflow):
    # Do workflow
    # Then save trace to database along with ensuring the target returned also has the current session id
    def __init__(self, mcp_tools=None):
        super().__init__()
        self.mcp_tools = mcp_tools

        self.main_expression_agent: Agent = Agent(
            name="MainExpressionAgent",
            model=OpenAIChat("gpt-4.1"),
            tools=[self.mcp_tools],
            description=dedent("""\
            You are the MainExpressionAgent, responsible for identifying suitable protein targets for Alzheimer's disease. Your job is to ensure that the targets 
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

            3. **Module and Cell-Type Annotation 🧠**  
            - Use `get_module_number_and_cell_type` to extract the **biological module** and **relevant cell type**.
            - Prioritize genes with cell-type-specific roles and clearly assigned module numbers.

            4. **Tissue Toxicity Filter ⚠️**  
            - Use `tissue_expression_tool` to screen for **off-target expression**.
            - Disqualify genes that are **highly expressed** in critical non-brain tissues (e.g. heart, liver, kidney).

            5. **Target Selection 🎯**  
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
                - Target DEG Cell (cell type)
                - Target Module Number (biological module)
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

    async def arun(
        self,
        uniprot_id: str,
        biological_mechanism: str, 
        citations: str,
        DEG_cell: str,
        module_number: str,
        in_silico_passed: bool
    ) -> AsyncIterator[RunResponse]:
        
        # Initialize MCP tools if not already done
        if not self.mcp_tools:
            async with MCPTools(url=SERVER_URL, transport="streamable-http") as mcp_tools:
                self.mcp_tools = mcp_tools

        # 1. Select target candidates
        print("Step 1: Selecting target candidates")
        self.mcp_tools.include_tools = ["tissue_expression_tool", "search_pubmed", "get_pubmed_abstract", "get_related_articles", "find_by_author", "get_module_number_and_cell_type"]
        async for response in await self.main_expression_agent.arun(dedent(f"""Target ID: {uniprot_id}
        Target Mechanism of Action: {biological_mechanism}
        Target Citations: {citations}
        Target DEG Cell: {DEG_cell}
        Target Module Number: {module_number}
        Target In Silico Passed: {in_silico_passed}"""), stream=True):
            yield response
        
        # 2. add all to database, and send email for experiment
        print("Step 2: Adding data to database")
        
        # Yield a completion response
        yield RunResponse(run_id=self.run_id, content=RunEvent.workflow_completed)
        return