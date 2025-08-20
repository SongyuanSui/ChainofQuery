# Chain of Query
Code for paper [Chain-of-Query: Unleashing the Power of LLMs in SQL-Aided Table Understanding via Multi-Agent Collaboration](./ChainOfQuery_paper_preprint.pdf)

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

Note: This is a preprint version hosted on GitHub. Once the paper is available on arXiv, please cite the arXiv version.

```bibtex
@misc{sui2025coq,
  title={Chain-of-Query: Unleashing the Power of LLMs in SQL-Aided Table Understanding via Multi-Agent Collaboration},
  author={Sui, Songyuan and Liu, Hongyi and Liu, Serena and Li, Li and Choi, Soo-Hyun and Chen, Rui and Hu, Xia},
  note = {Preprint. Available at GitHub},
  year={2025},
  howpublished = {\url{https://github.com/SongyuanSui/ChainofQuery/blob/main/ChainOfQuery_paper_preprint.pdf}}
}
```

## Acknowledgement

We thank [Chain-of-Table](https://arxiv.org/abs/2401.04398) for releasing the [code](https://github.com/google-research/chain-of-table/tree/main), [MAG-SQL](https://arxiv.org/abs/2408.07930) for releasing the [code](https://github.com/LancelotXWX/MAG-SQL), and [OpenTab](https://arxiv.org/abs/2402.14361) for releasing the [code](https://github.com/amazon-science/llm-open-domain-table-reasoner).