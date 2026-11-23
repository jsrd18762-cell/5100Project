# GTEx input data

Git LFS stores the expression matrix as `gtex_RSEM_gene_tpm.gz`. Run `git lfs pull` in the repository if Git LFS has not downloaded the data.

From the repository root, restore the original input file before running the analysis.

```sh
python -c "import gzip, shutil; src = gzip.open('dataset/gtex_RSEM_gene_tpm.gz', 'rb'); dst = open('dataset/gtex_RSEM_gene_tpm', 'wb'); shutil.copyfileobj(src, dst); src.close(); dst.close()"
```
