# GTEx Autoencoder Interpretability

This project uses autoencoders to study how bottleneck neuron activations relate to tissue gene expression in GTEx. The analysis compares gene scores derived from the decoder Jacobian and from weight propagation. It also evaluates how batch normalization (BN), activation functions, and dropout affect the resulting neuron scores.

This directory contains the input files, analysis scripts, and selected result tables. An end-to-end run also requires the upstream outputs expected by each script, including model checkpoints, layer outputs, and Jacobian matrices. Paths below are relative to this directory. `model_*` and `tissue_*` denote numbered model and tissue entries.

## Input files

| Path | Contents and purpose |
| --- | --- |
| `dataset/gtex_RSEM_gene_tpm` | Tab-separated GTEx expression matrix with gene IDs in the first column and sample IDs in the remaining column headers. The preprocessing code interprets the stored values as `log2(TPM + 0.001)`. |
| `dataset/GTEX_phenotype` | Tab-separated sample annotations, including sample ID, detailed tissue, primary site, sex, donor ID, and cohort. Step 1 matches these annotations to the expression matrix by sample ID. |
| `gene_atlas/GTEx_Tissues_V8_2023.txt` | GTEx V8 tissue gene sets and their member genes. Step 5 uses the source sets to build its tissue reference library. |
| `input_data/gene_name_19594.csv` | List of 19,594 gene-name entries used to select features for the reduced-input condition in Step 8. |
| `input_data/gene_name_60498.csv` | List of 60,498 gene-name entries aligned to the full expression feature order. |
| `input_data/tissue_appear_times.csv` | Tissue names and sample counts. |
| `input_data/tsne_label.csv` | Tissue labels indexed by sample position in the expression matrix. |

## Step 1 · Expression preprocessing and baseline training

| Path | Purpose |
| --- | --- |
| `step1/code/compare_log_offsets.py` | Compares four log offsets on a common sample of expression values and generates density plots, boxplots, and distribution summaries. |
| `step1/code/step1_core_experiment.py` | Aligns expression and phenotype data, saves the sample split and feature order, converts expression values to `log2(TPM + 1)`, and trains 20 baseline autoencoders. |
| `step1/results/log_offset_comparison_metadata.json` | Records the input scale, candidate transformations, sampling settings, and random seed for the Python comparison. |
| `step1/results/log_offset_distribution_summary.csv` | Stores counts, quantiles, and other summary statistics for the Python comparison. |
| `step1/results/r_plots/log_offset_distribution_summary_R.csv` | Stores the corresponding distribution summaries from the R plotting workflow. |
| `step1/results/r_plots/plot_log_offsets_R_metadata.json` | Records the transformation and sampling settings used by that R workflow. |
| `step1/results/r_plots/python_summary_validation_R.json` | Reports the agreement between the R and Python summaries and the tolerance used for validation. |
| `step1/results/r_plots/sampled_values_R.rds` | Serialized R cache of the sampled expression values used for plotting. |

## Step 2 · Layer outputs

| Path | Purpose |
| --- | --- |
| `step2/code/step2_extract_layer_outputs.py` | Loads the trained baseline models, extracts per-sample linear layer and selected BN and activation outputs, and calculates tissue medians. The final decoder output is summarized by tissue. |

## Step 3 · Bottleneck visualization and tissue differences

| Path | Purpose |
| --- | --- |
| `step3/code/step3_latent_analysis.py` | Runs t-SNE with seed 222 on each model's 64-dimensional bottleneck and tests each neuron for differences across tissues using the Kruskal–Wallis test. Applies Benjamini–Hochberg correction within each model. |
| `step3/statistics/model_*/kw_results.csv` | Per-neuron test statistics, p-values, adjusted q-values, effect sizes, and candidate flags for each of the 20 models. |
| `step3/statistics/model_*/candidates.csv` | Neurons passing both `q < 0.05` and `epsilon² > 0.14` for exploratory Jacobian and enrichment analyses. |
| `step3/statistics/model_*/tissue_median_ranks.csv` | Tissue sample counts, median neuron activations, and tissue ranks for each neuron. |

## Step 4 · Decoder Jacobian

| Path | Purpose |
| --- | --- |
| `step4/code/step4_decoder_jacobian.py` | Differentiates decoder gene outputs with respect to candidate bottleneck activations at individual samples. Summarizes signed gradients by tissue using means, standard deviations, standard errors, medians, and positive and negative fractions. Also saves a small set of individual Jacobians and the corresponding mappings. |

## Step 5 · Tissue reference sets and Jacobian enrichment

| Path | Purpose |
| --- | --- |
| `step5/code/step5_gsea.py` | Builds tissue reference sets, merges same-name gene features by averaging their Jacobian scores, prepares full ranked gene lists, and summarizes enrichment results. |
| `step5/code/00_setup_environment.R` | Sets up the local R package library and records or checks the package version lock. |
| `step5/code/00_reset_gsea_results.R` | Clears existing Step 5 enrichment outputs and completion records when explicitly enabled for a full rerun. |
| `step5/code/01_run_fgsea_parallel.R` | Runs or checks parallel `fgseaMultilevel` jobs grouped by model and source tissue. |
| `step5/code/02_plot_enrichment.R` | Draws full-ranking enrichment curves for the six selected display tissues and checks the plotted enrichment scores. |
| `step5/code/_step5_common.R` | Shared R functions for version checks, input validation, score loading, ranking, and enrichment. Steps 7 and 8 reuse this file. |
| `step5/gsea_results/model_*/tissue_*.csv.gz` | Compressed gene set enrichment analysis (GSEA) results for each model and source tissue against the reference sets. Columns include enrichment score (ES), normalized enrichment score (NES), p-value, adjusted p-value, and leading-edge genes. There are 1,080 result tables across 20 models and 54 source tissues. |
| `step5/gsea_results/model_*/tissue_*_warnings.csv` | Companion tables recording warnings for each model and source tissue. |

## Step 6 · Jacobian-based NGCI

NGCI is the signed Pearson correlation between a neuron's median activation and the matching tissue's GSEA ES across 53 reference-matched tissues.

| Path | Purpose |
| --- | --- |
| `step6/code/step6_ngci.py` | Pairs tissue activation medians with Jacobian-based ES values, computes two-sided Pearson statistics by neuron, and generates a display scatterplot. |
| `step6/statistics/ngci_tissue_pairs.csv` | Tissue-level pairs of median activation and ES, with model, neuron, and tissue identifiers. |
| `step6/statistics/ngci_per_neuron.csv` | Per-neuron NGCI, Pearson p-value, number of tissues, and calculation status. |
| `step6/statistics/ngci_model_summary.csv` | Model-level summaries of the NGCI distribution and counts of valid and undefined correlations. |

## Step 7 · Weight propagation and comparison with the Jacobian

| Path | Purpose |
| --- | --- |
| `step7/code/step7_weight_analysis.py` | Propagates signed bottleneck weights through the decoder to obtain one gene score vector per model and neuron. Prepares full ranked lists for GSEA, calculates weight-based NGCI, and compares gene scores and NGCI with the Jacobian route. |
| `step7/code/01_run_fgsea_parallel.R` | Runs or checks parallel enrichment for the gene rankings derived from weight propagation. |
| `step7/code/02_plot_enrichment.R` | Draws the six-tissue enrichment curve comparison for the main Step 7 workflow. |
| `step7/code/display_muscle.py` | Assembles the fixed display case `model_11 / unit_00014` from existing results. Prepares tables, an activation boxplot, NGCI scatterplots, and inputs for the display curves. |
| `step7/code/03_plot_muscle_enrichment.R` | Checks the saved ES values and draws the Jacobian and two-method enrichment curves for the fixed display case. |
| `step7/code/_step7_common.R` | Connects the shared Step 5 R functions to Step 7 inputs, reference sets, and output paths. |

## Step 8 · Model components and architecture

| Path | Purpose |
| --- | --- |
| `step8/code/step8_experiments.py` | Defines the BN, activation function × BN order, dropout × BN, and architecture experiments. Manages resumable training, prepares weight-derived gene rankings, computes weight-based NGCI, and fits linear mixed models (LMMs). |
| `step8/code/streamed_training.py` | Supports deep model training with limited GPU memory. It stages layer weights from CPU memory and stores gradients and Adam optimizer states on disk. It also includes numerical and checkpoint checks. |
| `step8/code/01_run_fgsea_parallel.R` | Runs or checks enrichment by model and experimental condition for the Step 8 ranked inputs. |
| `step8/code/_step8_common.R` | Shared Step 8 R functions for loading condition-specific scores and references and coordinating enrichment outputs. |

The completed LMM results are grouped under `step8/statistics/bn/`, `step8/statistics/activation_bn/`, and `step8/statistics/dropout_bn/`. In the patterns below, `<experiment>` refers to one of these three directory names.

| Path | Contents |
| --- | --- |
| `step8/statistics/<experiment>/fixed_effects.csv` | Fixed-effect coefficients on the Fisher-z scale, standard errors, confidence intervals, p-values, and BH-adjusted q-values. |
| `step8/statistics/<experiment>/lmm_input.csv` | Neuron-level model input, including the raw NGCI, the value used for fitting, and any replacement flags. |
| `step8/statistics/<experiment>/lmm_summary.txt` | Text summary of the fitted mixed model. |
| `step8/statistics/<experiment>/lmm_diagnostics.json` | Fit settings, convergence, variance estimates, and recorded warnings. |
| `step8/statistics/<experiment>/model_summaries.csv` | Per-model NGCI medians, quartiles, valid counts, and the fraction of valid neurons with `r > 0.5`. |
| `step8/statistics/{activation_bn,dropout_bn}/interaction_means.csv` | Fitted values for each combination of experimental factors, with confidence intervals on the Fisher-z and correlation scales. |
| `step8/statistics/structure/structure_lmm_results.csv` | Historical depth and input-size comparisons. The `source_file` column points to the earlier project, and this directory contains those reference estimates only. |

The 21 condition directories under `step8/statistics/conditions/` include the reused baseline, the no-BN condition, alternate activation/BN orders, and five dropout rates with and without BN.

| Path | Contents |
| --- | --- |
| `step8/statistics/conditions/*/matched_gsea.csv` | Tissue-matched enrichment results, including ES, NES, p-values, and adjusted q-values. |
| `step8/statistics/conditions/*/ngci_per_neuron.csv` | Weight-based NGCI and Pearson statistics by condition, model, and neuron. |
| `step8/statistics/conditions/*/stability_per_model.csv` | Within-model summaries of significant enrichment by reference tissue. |
| `step8/statistics/conditions/*/stability_across_models.csv` | Across-model frequencies and summaries of tissue enrichment. |

## Python bytecode caches

`step1/code/__pycache__/step1_core_experiment.cpython-313.pyc`, `step7/code/__pycache__/step7_weight_analysis.cpython-314.pyc`, and `step8/code/__pycache__/{step8_experiments,streamed_training}.cpython-313.pyc` are interpreter-generated bytecode copies of their corresponding Python modules.
