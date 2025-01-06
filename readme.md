这是openrank杯2024比赛，This is AI的仓库。

参赛题目：评估开源数据集的长期使用模式：一种引用网络视角

参赛选手：张雨昂 张雨欣 彭佳恒 

指导老师：王伟

**1.介绍**

**研究背景** 

数据集的评估是学术任务评价的基础和核心，评估数据集的使用模式对于选择合适的数据集以及推动相关科学研究任务的开展至关重要。研究人员在选择合适的数据集时常常面临挑战，主要因为他们对这些数据集的使用方式缺乏深入了解。
许多著名的开源数据集虽然已经非常成熟，其官方网站和相应的Github仓库多年未更新，但仍然被大量研究人员广泛使用。

**当前挑战** 

下面是这是两大著名的数据集的官网截图示例，官网基本上只提供了下载数据集的链接地址等少数信息。从官网上我们很难分析出其近期的被使用情况，以及长期的使用使用模式。

（1）IMDB

![image](https://github.com/user-attachments/assets/0fdda98b-178b-46a3-b526-6da6a088e3d8)

（2）ImageNet

![image](https://github.com/user-attachments/assets/4aad8506-8e51-455c-aa07-58e720bdfabe)

这是其对应的Github仓库截图示例，可以看出，这些数据集的相关文件已经很长时间没有更新了，其仓库背后的Github行为数据也寥寥无几。

（1）IMDB

![image](https://github.com/user-attachments/assets/25d34912-d021-4431-8390-c0fb72c340e4)

（2）ImageNet

![image](https://github.com/user-attachments/assets/8f1571a5-7a86-42af-b5ad-f77bdac4377d)

**常见的Github洞察指标举例**

常见的Github日志数据洞察指标：如活跃度、Issue数、star数、OpenRank值等等，背后都需要Github行为数据支撑：

![image](https://github.com/user-attachments/assets/a41d0967-03ca-4ff0-8daa-c7a7e0a57b20)

**研究动机**

（1）许多著名的开源数据集已经完善且多年未更新，但仍被大量研究人员广泛使用。

（2）其对应的官方网站上也没有提供被使用情况的相应信息，同时对应的GitHub仓库中也没有足够的行为活动数据以支撑我们使用传统的开源指标（活跃度、Issue数、star数、OpenRank值等等）来评估长期使用模式。

（3）但是我们发现，一般开源数据集都有一篇对应的学术论文（由数据集作者撰写）发表，可以**将数据集与其对应的学术论文的引用网络建立连接**，来分析此数据集的近期使用情况以及长期使用模式。

**2.方法与架构**

如下图：

（1）左侧：可以通过Paperwithcode网站获取这些开源数据集的模态/种类，每个开源数据集对应的Github仓库，数据集的名字。

（2）右侧：可以通过Semantic Ccholar网站，获取到这些开源数据集对应的学术论文，同时通过搜索其论文的引用网络，获取其被引用信息，来分析其长期使用模式。

![image](https://github.com/user-attachments/assets/fd799a37-f72b-4d2d-856a-759868a8316a)

**挑选具有代表性的数据集**

![image](https://github.com/user-attachments/assets/373c88bd-965f-4dd6-be0f-deb4697e4fac)

（1）我们获取了paperwithcode网站上Top5模态类型的数据集：分别是Image、Texts、Videos、Audio和Medical这五个模态类别。

（2）从这五种类型中，每一种类别分别挑选具有代表性的小、中、大规模的数据集。

（3）小规模数据集被定义其对应的学术论文被引用少于500次的数据集，中等规模的数据集是指引用次数在500到5000次之间的数据集，大规模数据集是指引用次数超过5000次的数据集。

**3.实验部分**

**数据规模**

![image](https://github.com/user-attachments/assets/ad83c897-351e-4988-8ef9-8744fe352d8a)

（1）除了使用OpenDigger的开源行为日志数据外，我们还通过上述方法补充了这些引用数据信息。

（2）补充数据量为：45965条。

（3）所有的引用信息均以开源至参赛仓库中：https://github.com/TIAI-ThisisAI/OpenRank2024

**实验结果**

**五种类别的15个开源数据集的总被引用量趋势**

![image](https://github.com/user-attachments/assets/e3827756-5863-4a93-b611-e5be4334b7c2)

（1）从累计引用量来看，Image-L也是最突出的一类。

（2）自2017年以来，其总被引量呈指数级增长，到2024年远远超过其他类别，这表明大规模图像数据集在研究领域占据中心和主导地位。

（3）同样，Text-L和Medical-L的累积被引量也快速上升，尤其是Text-L，其增长轨迹自2020年以来几乎与Image-L平行，表明大规模文本数据集与大规模图像数据集的差距正在逐渐缩小。

**五种类别的15个开源数据集的每年新增引用趋势**

![image](https://github.com/user-attachments/assets/a4d93591-e2da-420f-892b-1b137fe88384)

（1）大规模数据集，如Image-L、Text-L和Medical-L，显示出显著的引文增长，其中Image-L在2022年达到峰值，但保持了较高的引文数量。

（2）尽管近年来有所放缓，但在NLP和医疗人工智能的推动下，大规模图像和文本数据集仍在增长。

（3）与图像和文本数据集相比，Audio-L和Video-L数据集表现出更慢、更稳定的增长。

（4）中等规模的数据集，如Image-M和Text-M，显示引文以较慢的速度逐渐增加。

（5）小规模数据集（如音频和视频）最初出现增长，但在2020年后停滞或下降，反映出人们的兴趣降低。

**同一模态的不同规模数据集**

![image](https://github.com/user-attachments/assets/690db3d2-ddfd-44b6-a38a-c16af84fc2d9)

与中型（Image-M）和小型（Image-S）数据集相比，大型图像数据集（Image-L）的增长曲线明显更陡峭。Image-L 的累计引用量呈指数级增长，始于 2017 年左右，并在 2020 年后急剧加速。

到 2024 年，Image-L 的总引用量超过 8,000，远远超过 Image-M 和 Image-S。快速增长凸显了大型图像数据集的持续流行和影响力，这得益于它们在图像识别、目标检测等深度学习任务中的关键作用。

**项目贡献协作网络**

![image](https://github.com/user-attachments/assets/a942800f-bef2-4e98-95a6-b6bb509c0dbb)

（1）以image中的大型数据集对应的最高star数的Github仓库为例，分析了项目的贡献协作网络

（2）可以看出此数据集吸引了大量的有影响力的贡献者进行贡献，包括MarkDaoust, nealwu, cshtjn 等

（3）同时还吸引到了googlebot这种知名公司的机器人以及tensorflowbutler 等进行自动化协作，显示了项目中 CI/CD 流程和自动化协作的重要

**项目生态协作网络**

![image](https://github.com/user-attachments/assets/3165b137-bb4b-4120-b99a-37a3a16e416d)

（1）以image中的大型数据集对应的最高star数的Github仓库为例，分析了该仓库的项目生态协作网络

（2）可以看出此数据集吸引了大量的知名仓库之间的互相协作：如pytorch、微软的vscode、huggingface社区以及huggingface的transformers等等

（3）同时也可以看出越被大量引用的知名数据集和仓库，他们能更吸引到更知名的仓库和开发者来产生协作关系，从来带来“富俱乐部”效应

**项目社区协作网络**

![image](https://github.com/user-attachments/assets/6e4e198a-3902-4e2a-a43c-08f45f22f970)

（1）以image中的大型数据集对应的最高star数的Github仓库为例，分析了该仓库的项目社区协作网络

（2）可以看出此数据集吸引了大量知名的开发者和社区/公司，除了中国以外，还有德国、英国、美国、印度等大量的开发者和社区/公司与该仓库进行了协作

（3）同时还吸引到了google、nvidia、microsoft这种知名公司产生协作关系

**4.开源协作**

**开源协作 issue讨论**

本团队三个成员自从参加比赛开始就一直采取开源协作的方式，详情可见issue

从讨论选题开始就在github上以issue的形式进行讨论，共计4个issue，六十多条评论回复。

**开源协作 PR贡献**

本项目也是完全以开源协作的方式进行代码迭代，以共计31个PR的形式完成本项目的开发。






