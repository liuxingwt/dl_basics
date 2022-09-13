## 1.目标检测任务的损失函数分析

#### 1.1 CenterPoint的损失函数分析
| Item | Cls损失 | Reg损失 | Mask Pillar V1 | Mask Pillar V2 | Mask Pillar V3 |
| :-----:| :----: | :----: | :-----:| :-----:| :-----:|
| Target | HeatMap(Gaussion smooth) | box用mask处理 | HeatMap(Gaussion smooth) | HeatMap(Gaussion smooth) | HeatMap(Gaussion smooth) |
| Prediction | HeatMap(sigmoid处理) | box用mask处理 | Heat Map | HeatMap(sigmoid处理) | HeatMap(sigmoid处理) |
| Loss Fucntion| Fast Focal Loss(CE) | L1 loss | L1 loss | Fast Focal Loss(CE) | Fast Focal Loss(CE) |
| 损失函数注意项 | 负样本含中间态，正样本Fast处理| 只计算mask区域的损失 | 全局都计算了loss | 负样本含大部分中间态，正样本含少量中间态 | 负样本含中间态，正样本Fast处理|
| 备注 |mask+index确定正样本| None | None | 使用阈值划分正负样本 | mask+index确定正样本 |

#### 1.2 目标检测分类loss发展
+ （1）常规的前景和背景的交叉熵损失
+ （2）focal loss：把loss中well-classified的样本权重降低，此外还加了一个固定的负样本balance系数；
+ （3）CornerNet和CenterNet：把上面固定的负样本系数改为和gt相关的多项式；对正样本使用了fast focal loss的实现；
+ （4）CenterPoint：同上，对gt使用了高斯平滑处理；
