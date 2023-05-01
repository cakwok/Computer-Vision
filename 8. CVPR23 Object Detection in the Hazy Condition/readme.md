![image](https://user-images.githubusercontent.com/21034990/235517053-4ad76849-efc8-4881-973f-c7c4fcde1e04.png)

### Object Detection in Haze
The objective of the research work is to participate in the CVPR2023 UG2 challenge - Object Detection in Haze.

Scene understanding for applications such as intelligence, surveillance, and reconnaissance (ISR) and autonomous vehicles becomes extremely challenging in adverse weather conditions such as haze, fog and mist. These atmospheric phenomena with smoke particles and minute water droplets often result in imagery with non-linear noise, blur, reduced contrast levels and color dimming issues. Thus, the on-board vision systems are significantly obscured. These visual artifacts generated from uncontrolled and potentially changing environment or Degraded Visual Environments (DVE) poses major challenges in image enhancement and restoration, and object detection and classification, some of the key tasks towards the final goal of semantic scene understanding.  (quoted from UG2 + Challenge)

The challenge aims to evaluate and advance object detection algorithms’ robustness on images captured from hazy environmental situations.

#### Dataset
Dataset is based on the A2I2-Haze, the first real haze dataset with in-situ smoke measurement aligned to aerial imagery. A2I2-Haze has paired haze and haze-free imagery that will allow fine-grained evaluation of computer vision algorithms.  A total of 229 paired hazy/clean frame images extracted from 12 videos. 

<img src = "https://user-images.githubusercontent.com/21034990/235517399-4dd6896d-28d7-41a2-8b04-606b56fc324b.png" width = 500> <img src = "https://user-images.githubusercontent.com/21034990/235517473-bdac3a73-7ece-42bb-851c-5b759ebce24b.png" width = 500><br>

#### Results
Using YOLOv8 as the object detection framework, the model has achieved 0.994 mAP50 and 0.933 mAP50-95.
<img src = "https://user-images.githubusercontent.com/21034990/235518520-5bf91179-6947-43d5-82ef-1e9455c895cb.png" width = 700>
<img src = "https://user-images.githubusercontent.com/21034990/235518374-397f53e4-c3b0-4864-a46e-73a26f30cfbb.png" width = 700>
