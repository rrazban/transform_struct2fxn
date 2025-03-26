As shown in `results.png`, our transformer architecture approach outperforms a previously published deep neural network approach. We obtain an average Pearson correlation between predicted and empirical functional connectivities of 0.651 ± 0.096, compare to 0.507 ± 0.200 for the DNN approach. The central dashed line in each distribution corresponds to the median, while neighboring dashed lines correspond to lower and upper quartiles, respectively.

An important distinction is that our transformer approach does not attempt to maximally capture individual differences. This could artificially inflate our performance by only capturing average effects among the 1010 Human Connectome Project individuals.


|   | our transformer |DNN (Sarwar 2021) |
| :---------------- | :------: | :----: |
| <r(predicted,empirical)>   |   0.65 ± 0.1  | 0.51 ± 0.2 |
|  |  |  |
| Num fitted params         |   388,949,150   | 13,441,438 |
| Num epochs trained   |   1,000   | 20,000 |
| Total memory    |   14.6GB   | 0.7GB |
| Maximize individuality     |   no   | yes |


References:
* Sarwar, T., Tian, Y., Yeo, B.T., Ramamohanarao, K. and Zalesky, A., 2021. Structure-function coupling in the human connectome: A machine learning approach. NeuroImage, 226, p.117609.
* Zalesky, A., Sarwar, T., Tian, Y., Liu, Y., Yeo, B.T. and Ramamohanarao, K., 2024. Predicting an individual’s functional connectivity from their structural connectome: Evaluation of evidence, recommendations, and future prospects. Network Neuroscience, 8(4), pp.1291-1309.
