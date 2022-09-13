## 目标检测任务的损失函数分析

#### CenterPoint的损失函数分析
| Item | Cls损失 | Reg损失 | Mask Pillar V1 | Mask Pillar V2 |
| :-----:| :----: | :----: | :-----:| :-----:|
| Target | HeatMap(Gaussion smooth) | box用mask处理 | HeatMap(Gaussion smooth) | HeatMap(Gaussion smooth) |
| Prediction | HeatMap(sigmoid处理) | box用mask处理 | Heat Map | HeatMap(sigmoid处理) |
| Loss Fucntion| Fast Focal Loss(CE) | L1 loss | L1 loss | Focal Loss(CE) |
| 损失函数注意项 | 对正样本做Fast处理| 只计算mask区域的损失 | 全局都计算了loss | 负样本包含中间态，正样本 |
| 备注 |

