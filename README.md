# [NeurIPS 2026] SimpleEvol: An Agent-Loop Framework for LLM-Driven Automated Heuristic Design with Minimal Human Priors

## 1. Framework Overview

SimpleEvol is a lightweight LLM-driven agent-loop framework for automatic heuristic design that maintains a single evolving search trajectory with minimal human scaffolding around the LLM for the heuristic search. At each iteration, the LLM generates a candidate heuristic and a concise description, while an evaluator measures its performance on training instances and returns objective values, runtime information, and any execution errors. Periodic history compression turns the accumulated feedback into compact working notes and retains the best-so-far heuristic in the prompt context, allowing the agent to build on previous discoveries without a separate planner or reflection module. This design shifts the emphasis from handcrafted search operators to a lightweight information interface that supports the LLM's own heuristic exploration.

![SimpleEvol framework: LLM generation, heuristic evaluation, feedback, and periodic history compression](figures/main_frame.png)

*SimpleEvol combines a generation-and-evaluation agentic loop with best-so-far retention and compressed search history.*

## 2. Advantages over Existing LLM-based AHD Frameworks

Under a common budget for heuristic evaluations, SimpleEvol combines a simpler search framework with strong performance across ten backbone LLMs. It achieves the highest intelligence conversion efficiency (ICE, the fitted slope of performance against the model-intelligence proxy) among these four methods on both TSP Constructive and CVRP-ACO: **2.1941 on TSP** and **6.2128 on CVRP**, compared with **1.8174** and **5.4381** for the strongest baseline by ICE, FunSearch. The model-wise comparisons below show that this advantage is also reflected in direct performance rankings aross different backbones.

Each column compares the four methods using the same backbone. The upper-left label gives the rank by average gap, while the percentage and color show relative gap improvement over the four-method mean for that backbone. Positive values indicate a lower, better gap than that mean.

**TSP Constructive.** SimpleEvol ranks first on **6 of 10 backbones** and among the top two on **9 of 10**, the most first-place finishes among the four methods. With GPT-5-mini, it achieves a **16.9% relative gap improvement over the four-method mean**. The paper additionally reports the lowest test gap among the four methods at all three tested sizes with this backbone: **4.77%, 6.47%, and 9.49%** for N=50, 100, and 200, respectively.

![TSP Constructive: model-wise performance ranks and relative gap improvements for SimpleEvol, FunSearch, EoH, and ReEvo](figures/tsp_advantage_rank_01.png)

**CVRP-ACO.** SimpleEvol ranks first on **5 of 10 backbones** and among the top two on **8 of 10**, again achieving the most first-place finishes. Its relative gap improvement over the four-method mean reaches **24.7% with Qwen3-Instruct** and **14.7% with GPT-5-mini**. With GPT-5-mini, it also achieves the lowest test gap at every tested size: **0.34%, 4.31%, and 4.26%** for N=50, 100, and 200. Collectively, these results demonstrate competitive model-wise performance and stronger fitted scaling within the evaluated tasks and backbones.

![CVRP-ACO: model-wise performance ranks and relative gap improvements for SimpleEvol, FunSearch, EoH, and ReEvo](figures/cvrp_advantage_rank_01.png)

This README also describes the current implementation and configuration.

## 3. Installation

Run commands from the directory containing `main.py` and `requirements.txt`. The project uses Python 3.10+ syntax. Linux is recommended for experiments because evaluator timeouts use Unix signals.

```bash
python -m pip install -r requirements.txt
```

Dependencies include NumPy, SciPy, PyTorch, Numba, Hydra, tqdm, and the OpenAI SDK. A GPU is not required. 

## 4. Configure the Task, Algorithm, and LLM

Edit `cfg/config.yaml` to select the task, search algorithm, and LLM client:

```yaml
defaults:
  - _self_
  - problem: tsp_constructive  # tsp_constructive, cvrp_aco, or fssp_gls
  - llm_client: openrouter  # openai or openrouter
  - baseline: funsearch
  - override hydra/output: local

# Select the search algorithm.
algorithm: simple_evol  # simple_evol or funsearch
output_root: ${algorithm}_outputs
```

`algorithm` switches between SimpleEvol and FunSearch; `llm_client` only selects the API client. The `baseline: funsearch` entry loads FunSearch settings but does not select the active algorithm. 

Set the model and API-key reference in the corresponding client configuration.

**OpenAI** (`cfg/llm_client/openai.yaml`):

```yaml
model: gpt-4o-mini
api_key: ${oc.env:OPENAI_API_KEY}
```

**OpenRouter** (`cfg/llm_client/openrouter.yaml`):

```yaml
model: openai/o3-mini
api_key: ${oc.env:OPENROUTER_API_KEY}
```

Keep the other fields unchanged. Before running, set the environment variable for your selected provider:

```bash
# Linux / macOS: use OPENAI_API_KEY for the OpenAI client
export OPENROUTER_API_KEY="YOUR_API_KEY"
```

```powershell
# Windows PowerShell: use OPENAI_API_KEY for the OpenAI client
$env:OPENROUTER_API_KEY = "YOUR_API_KEY"
```

Run from the project root:

```bash
python main.py
```


## 5. Run the supported problems

Set `problem` in the `defaults` list of `cfg/config.yaml` to one of the tasks below. After configuring the algorithm and client in Section 4 and setting the API-key environment variable, run:

```bash
python main.py
```

| Problem | Candidate function | Training data used by the default configuration | Final test data |
|---|---|---|---|
| `tsp_constructive` | `select_next_node` | 64 synthetic instances, N=50 | 64 instances per size, N=50, 100, 200 |
| `cvrp_aco` | `heuristics` | 10 synthetic instances, N=50 | 64 instances per size, N=50, 100, 200 |
| `fssp_gls` | `get_matrix_and_jobs` | 16 synthetic instances, 50 jobs | All 110 Taillard instances in `TestingData/` |

### Search settings

Defaults in `cfg/config.yaml`:

```yaml
max_experiments: 820
compress_every: 5
enable_test_eval: False
timeout: 60
```

- `max_experiments`: number of generated candidates evaluated by the main loop. Summaries and format retries require additional calls. FSSP also evaluates its seed as experiment 0.
- `compress_every`: number of evaluated candidates between context compressions. Zero disables compression.
- `enable_test_eval`: enables intermediate test evaluations in addition to training. Keep this disabled for held-out testing: when enabled, test scores are included in the current feedback implementation. Not used in the results of our framework.
- `timeout`: evaluation time budget in seconds.


For a shorter SimpleEvol exploratory run, temporarily set the following in `cfg/config.yaml`, then run `python main.py`:

```yaml
max_experiments: 5
compress_every: 5
```

This still calls a real LLM and performs final testing. It is not an offline test.


## 6. Datasets and Outputs.

To generate datasets manually from the project root:

```bash
python -m problems.tsp_constructive.gen_inst
python -m problems.cvrp_aco.gen_inst
python -m problems.fssp_gls.gen_inst
```

Hydra stores each run under:

```text
simple_evol_outputs/<problem_name>-<problem_type>/<date>_<time>/
    .hydra/
    <job-name>.log
    experiment_records/
        exp_001.txt
        ...
```

FSSP also records `exp_000.txt` for the seed. Each experiment record contains the candidate code, description, objective, timing, and errors. The final selected code is saved in `problems/<problem>/best_candidates/`. Final test scores are logged by `main.py`.


