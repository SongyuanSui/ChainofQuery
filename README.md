# Chain of Query
[AACL 2025 Main Conference (Oral)] This is the official code for the paper [Chain-of-Query: Unleashing the Power of LLMs in SQL-Aided Table Understanding via Multi-Agent Collaboration](https://aclanthology.org/2025.ijcnlp-long.53/)

<img src="./assets/overview.png" align="middle" width="95%">

## Environment

```shell
conda create --name coq python=3.10 -y
conda activate coq
pip install -r requirements.txt
```

## Command Usages

### Arguments

- `--model`: name of the LLM
- `--num_samples`: number of the first n samples to evaluate

### Example usages

1. Run tests on the first 20 cases with gpt-3.5-turbo-1106.

   ```shell
    python run_chain_of_query.py --model gpt-3.5-turbo-1106 --num_samples 20
   ```

2. Evaluate the baseline MAG-SQL.

   ```shell
    python run_mag_sql.py --model gpt-3.5-turbo-1106 --num_samples 20
   ```

3. Evaluate the baseline Chain-of-Table.

   ```shell
    python run_chain_of_table.py --model gpt-3.5-turbo-1106 --num_samples 20
   ```

## Cite

```bibtex
@inproceedings{sui-etal-2025-chain,
    title = "Chain-of-Query: Unleashing the Power of {LLM}s in {SQL}-Aided Table Understanding via Multi-Agent Collaboration",
    author = "Sui, Songyuan  and
      Liu, Hongyi  and
      Liu, Serena  and
      Li, Li  and
      Choi, Soo-Hyun  and
      Chen, Rui  and
      Hu, Xia",
    editor = "Inui, Kentaro  and
      Sakti, Sakriani  and
      Wang, Haofen  and
      Wong, Derek F.  and
      Bhattacharyya, Pushpak  and
      Banerjee, Biplab  and
      Ekbal, Asif  and
      Chakraborty, Tanmoy  and
      Singh, Dhirendra Pratap",
    booktitle = "Proceedings of the 14th International Joint Conference on Natural Language Processing and the 4th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics",
    month = dec,
    year = "2025",
    address = "Mumbai, India",
    publisher = "The Asian Federation of Natural Language Processing and The Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.ijcnlp-long.53/",
    pages = "957--986",
    ISBN = "979-8-89176-298-5",
    abstract = "Table understanding requires structured, multi-step reasoning. Large Language Models (LLMs) struggle with it due to the structural complexity of tabular data. Recently, multi-agent frameworks for SQL generation have shown promise in tackling the challenges of understanding tabular data, but existing approaches often suffer from limitations such as the inability to comprehend table structure for reliable SQL generation, error propagation that results in invalid queries, and over-reliance on execution correctness. To address these issues, we propose Chain-of-Query (CoQ), a novel multi-agent framework for SQL-aided table understanding. CoQ adopts natural-language-style representations of table schemas to abstract away structural noise and enhance understanding. It employs a clause-by-clause SQL generation strategy to improve query quality and introduces a hybrid reasoning division that separates SQL-based mechanical reasoning from LLM-based logical inference, thereby reducing reliance on execution outcomes. Extensive experiments across four models and five widely used benchmarks demonstrate that CoQ achieves substantial accuracy improvements and significantly lowers invalid SQL rates compared to prior generic LLM-based, SQL-aided, and hybrid baselines, confirming its superior effectiveness in table understanding. The code is available at https://github.com/SongyuanSui/ChainofQuery."
}
```

## Acknowledgement

We thank [Chain-of-Table](https://arxiv.org/abs/2401.04398) for releasing the [code](https://github.com/google-research/chain-of-table/tree/main), [MAG-SQL](https://arxiv.org/abs/2408.07930) for releasing the [code](https://github.com/LancelotXWX/MAG-SQL), and [OpenTab](https://arxiv.org/abs/2402.14361) for releasing the [code](https://github.com/amazon-science/llm-open-domain-table-reasoner).