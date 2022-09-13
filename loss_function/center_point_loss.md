## 1.目标检测任务的损失函数分析

#### 1.1 CenterPoint的损失函数分析
| Item | Cls损失 | Reg损失 | Mask Pillar V1 | Mask Pillar V2 | Mask Pillar V3 |
| :-----:| :----: | :----: | :-----:| :-----:| :-----:|
| Target | HeatMap(Gaussion smooth) | box用mask处理 | HeatMap(Gaussion smooth) | HeatMap(Gaussion smooth) | HeatMap(Gaussion smooth) |
| Prediction | HeatMap(sigmoid处理) | box用mask处理 | Heat Map | HeatMap(sigmoid处理) | HeatMap(sigmoid处理) |
| Loss Fucntion| Fast Focal Loss(CE) | L1 loss | L1 loss | Fast Focal Loss(CE) | Fast Focal Loss(CE) |
| 损失函数注意项 | 负样本含中间态，正样本Fast处理| 只计算mask区域的损失 | 全局都计算了loss | 负样本含大部分中间态，正样本含少量中间态 | 负样本含中间态，正样本Fast处理|
| 备注 |mask+index确定正样本| None | None | 使用阈值划分正负样本 | mask+index确定正样本 |

