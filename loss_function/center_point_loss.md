## 目标检测任务的损失函数分析

#### CenterPoint的损失函数分析
| - | Cls损失 | Reg损失 | Mask PIllar V1 |
| :-----:| :----: | :----: | :-----:|
| Ground Truth | heat map(Gaussion smooth) |  | Seg的分类思路 |
| Prediction | 用sigmoid预处理 |  | 用sigmoid预处理 |
| 损失函数| Cross Entropy | L1 loss | CE loss |
| 备注| focal loss，fast正样本| 只处理正样本的mask区域 |  |  |
