# GlanceVAD: Exploring Glance Supervision for Label-efficient Video Anomaly Detection

<p align="center">
<img src="GlanceVAD.png" >
  </p>


> **Abstract:**
> In recent years, video anomaly detection has been extensively investigated in both unsupervised and weakly supervised settings to alleviate costly temporal labeling. Despite significant progress, these methods still suffer from unsatisfactory results such as numerous false alarms, primarily due to the absence of precise temporal anomaly annotation. In this paper, we present a novel labeling paradigm, termed "glance annotation", to achieve a better balance between anomaly detection accuracy and annotation cost. Specifically, glance annotation is a random frame within each abnormal event, which can be easily accessed and is cost-effective. To assess its effectiveness, we manually annotate the glance annotations for two standard video anomaly detection datasets: UCF-Crime and XD-Violence. Additionally, we propose a customized **GlanceVAD** method, that leverages gaussian kernels as the basic unit to compose the temporal anomaly distribution, enabling the learning of diverse and robust anomaly representations from the glance annotations. Through comprehensive analysis and experiments, we verify that the proposed labeling paradigm can achieve an excellent trade-off between annotation cost and model performance. Extensive experimental results also demonstrate the effectiveness of our GlanceVAD approach, which significantly outperforms existing advanced unsupervised and weakly supervised methods. Code and annotations will be publicly available.

> **Motivation:**
> Our key insight is to leverage anomaly video data, which is harder to collect compare with normal videos, through extremly cost-efficient glance annotation (one frame click during abnormal events). The reduced bias toward the anomaly context results in the siginificant performance improvement, which provide a new practical labeling paradigm for Video Anomaly Detection.
<p align="center">
<img src="motivation.png" >
  </p>

## 🆕:Updates
- (2024-03-08) Comming soon.

## 📝:Results
We use Area Under the Curve (AUC) of the frame-level Receiver Operating Characteristic (ROC) as the evaluation metric for UCF-Crime, and AUC of the frame-level precision-recall curve (AP) is utilized for XD-Violence as the standard evaluation metric.
we also evaluate the AUC/AP of abnormal videos (termed by AUC_A/AP_A)

|Method | Dataset  | Feature| AUC | AUC_A |
| ----- | -----    | ----- |----- | ----- |
|UR-DMU(baseline) |UCF-Crime | I3D   | 86.97 | 70.81 |
|**GlanceVAD**(Ours)|UCF-Crime | I3D   | **91.96** | **84.94** |

|Method | Dataset  | Feature|  AUC | AUC_A |
| ----- | -----    | ----- | ----- |----- |
|UR-DMU(baseline)| XD-Violence | I3D | 81.66 | 83.51 |
|**GlanceVAD**(Ours) | XD-Violence | I3D | **89.40** | **89.85** |

## 📊:Qualitative Results
<p align="center">
<img src="quality.png" >
  </p>

##  🛰️:References
We apreciate the repos below for the codebase.

- [UR-DMU](https://github.com/henrryzh1/UR-DMU)
- [S3R](https://github.com/louisYen/S3R)
- [RTFM](https://github.com/tianyu0207/RTFM)
