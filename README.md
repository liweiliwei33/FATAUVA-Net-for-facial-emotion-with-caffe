# FATAUVA-Net-for-facial-emotion-detection-with-caffe
This is an implementation of FATAUVA-Net model in paper: An Integrated Deep Learning Framework for Facial Attribute Recognition, Action Unit Detection and Valence-Arousal Estimation


网络架构：

Core Layer 使用 multi-task CNN 参考 [29]
Attribute Layer 首先crop face area 参考 [41]，如眼眉嘴；从CelebA数据库中选择10个属性；
网络结构如下：
