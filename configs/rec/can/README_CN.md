[English]() | 中文

# CAN (Counting-Aware Network)
<!--- Guideline: use url linked to abstract in ArXiv instead of PDF for fast loading.  -->

> [CAN: When Counting Meets HMER: Counting-Aware Network for Handwritten Mathematical Expression Recognition](https://arxiv.org/pdf/2207.11463.pdf)

## 1. 模型描述
<!--- Guideline: Introduce the model and architectures. Cite if you use/adopt paper explanation from others. -->

CAN是具有一个弱监督计数模块的注意力机制编码器-解码器手写数学公式识别算法。本文作者通过对现有的大部分手写数学公式识别算法研究，发现其基本采用基于注意力机制的编码器-解码器结构。该结构可使模型在识别每一个符号时，注意到图像中该符号对应的位置区域，在识别常规文本时，注意力的移动规律比较单一（通常为从左至右或从右至左），该机制在此场景下可靠性较高。然而在识别数学公式时，注意力在图像中的移动具有更多的可能性。因此，模型在解码较复杂的数学公式时，容易出现注意力不准确的现象，导致重复识别某符号或者是漏识别某符号。

针对于此，作者设计了一个弱监督计数模块，该模块可以在没有符号级位置注释的情况下预测每个符号类的数量，然后将其插入到典型的基于注意的HMER编解码器模型中。这种做法主要基于以下两方面的考虑：1、符号计数可以隐式地提供符号位置信息，这种位置信息可以使得注意力更加准确。2、符号计数结果可以作为额外的全局信息来提升公式识别的准确率。

<p align="center">
  <img src="https://temp-data.obs.cn-central-221.ovaijisuan.com/mindocr_material/miss_word.png" width=640 />
</p>
<p align="center">
  <em> 图1. 手写数学公式识别算法对比 [<a href="#参考文献">1</a>] </em>
</p>

CAN模型由主干特征提取网络、多尺度计数模块（MSCM）和结合计数的注意力解码器（CCAD）构成。主干特征提取通过采用DenseNet得到特征图，并将特征图输入MSCM，得到一个计数向量（Counting Vector），该计数向量的维度为1*C，C即公式词表大小，然后把这个计数向量和特征图一起输入到CCAD中，最终输出公式的latex。

<p align="center">
  <img src="https://temp-data.obs.cn-central-221.ovaijisuan.com/mindocr_material/total_process.png" width=640 />
</p>
<p align="center">
  <em> 图2. 整体模型结构 [<a href="#参考文献">1</a>] </em>
</p>

多尺度计数模MSCM块旨在预测每个符号类别的数量，其由多尺度特征提取、通道注意力和池化算子组成。由于书写习惯的不同，公式图像通常包含各种大小的符号。单一卷积核大小无法有效处理尺度变化。为此，首先利用了两个并行卷积分支通过使用不同的内核大小（设置为 3×3 和 5×5）来提取多尺度特征。在卷积层之后，采用通道注意力来进一步增强特征信息。

<p align="center">
  <img src="https://temp-data.obs.cn-central-221.ovaijisuan.com/mindocr_material/MSCM.png" width=640 />
</p>
<p align="center">
  <em> 图3. MSCM多尺度计数模块 [<a href="#参考文献">1</a>] </em>
</p>

结合计数的注意力解码器：为了加强模型对于空间位置的感知，使用位置编码表征特征图中不同空间位置。另外，不同于之前大部分公式识别方法只使用局部特征进行符号预测的做法，在进行符号类别预测时引入符号计数结果作为额外的全局信息来提升识别准确率。

<p align="center">
  <img src="https://temp-data.obs.cn-central-221.ovaijisuan.com/mindocr_material/CCAD.png" width=640 />
</p>
<p align="center">
  <em> 图4. 结合计数的注意力解码器CCAD [<a href="#参考文献">1</a>] </em>
</p>

## 参考文献
<!--- Guideline: Citation format GB/T 7714 is suggested. -->
[1] Xiaoyu Yue, Zhanghui Kuang, Chenhao Lin, Hongbin Sun, Wayne Zhang. RobustScanner: Dynamically Enhancing Positional Clues for Robust Text Recognition. arXiv:2007.07542, ECCV'2020
