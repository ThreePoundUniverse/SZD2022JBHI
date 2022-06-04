# Continuous Seizure Detection Based on Transformer and Long-Term iEEG

## Abstract
Automatic seizure detection algorithms are necessary for patients with refractory epilepsy. Many excellent algorithms have achieved good results in seizure detection. Still, most of them are based on discontinuous intracranial electroencephalogram (iEEG) and ignore the impact of different channels on detection. This study aimed to evaluate the proposed algorithm using continuous, long-term iEEG to show its applicability in clinical routine. In this study, we introduced the ability of the transformer network to calculate the attention between the channels of input signals into seizure detection. We proposed an end-to-end model that included convolution and transformer layers. The model did not need feature engineering or format transformation of the original multi-channel time series. Through evaluation on two datasets, we demonstrated experimentally that the transformer layer could improve the performance of the seizure detection algorithm. For the SWEC-ETHZ iEEG dataset, we achieved 97.5\% event-based sensitivity, 0.06/h FDR, and 13.7 s latency. For the TJU-HH iEEG dataset, we achieved 98.1\% event-based sensitivity, 0.22/h FDR, and 9.9 s latency. In addition, statistics showed that the model allocated more attention to the channels close to the seizure onset zone within 20 s after the seizure onset, which improved the explainability of the model. This paper provides a new method to improve the performance and explainability of automatic seizure detection.

## Versions used in our code testing
Note that different versions won't likely cause dependencies issues.
```
tensorflow-gpu == 2.4.1 
```

## Contact
For any inconvernienes and bug reports, contact ```syuri@tju.edu.cn```

## Citation
Waiting for upload
