## 训练  
1.基础问题
a.样本问题，mtcnn训练时，会把训练的原图样本，通过目标所在区域进行裁剪，得到三类训练样本，即：正样本、负样本、部分(part)样本
其中：
裁剪方式：对目标区域，做平移、缩放等变换得到裁剪区域
IoU：目标区域和裁剪区域的重合度

此时三类样本如下定义：
正样本：IoU >= 0.65，标签为1
负样本：IoU < 0.3，标签为0
部分(part)样本：0.65 > IoU >= 0.4，标签为-1

b.网络问题，mtcnn分为三个小网络，分别是PNet、RNet、ONet，新版多了一个关键点回归的Net（这个不谈）。
PNet：12 x 12，负责粗选得到候选框，功能有：分类、回归
RNet：24 x 24，负责筛选PNet的粗筛结果，并微调box使得更加准确和过滤虚警，功能有：分类、回归
ONet：48 x 48，负责最后的筛选判定，并微调box，回归得到keypoint的位置，功能有：分类、回归、关键点

c.网络大小的问题，训练时输入图像大小为网络指定的大小，例如12 x 12，而因为PNet没有全连接层，是全卷积的网络，所以预测识别的时候是没有尺寸要求的，那么PNet可以对任意输入尺寸进行预测得到k个boundingbox和置信度，通过阈值过滤即可完成候选框提取过程，而该网络因为结构小，所以效率非常高。

2.训练步骤
参考：https://github.com/dlunion/mtcnn/tree/master/train 
一般训练几万次后，loss到0.0x的时候就可以接受了
记得在当前目录下创建models-12、models-24、models-48来迎接喜气招财哟~

# mtcnn-caffe
Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Neural Networks.

This project provide you a method to update multi-task-loss for multi-input source.
 
![result](https://github.com/CongWeilin/mtcnn-caffe/blob/master/demo/result.jpg)

## Requirement
0. Ubuntu 14.04 or 16.04
1. caffe && pycaffe: [https://github.com/BVLC/caffe](https://github.com/BVLC/caffe)
2. cPickle && cv2 && numpy 

## Train Data
The training data generate process can refer to [Seanlinx/mtcnn](https://github.com/Seanlinx/mtcnn)

Sample almost similar to Seanlinx's can be found in `prepare_data`

- step1. Download Wider Face Training part only from Official Website and unzip to replace `WIDER_train`

- step2. Run `gen_12net_data.py` to generate 12net training data. Besides, `gen_net_imdb.py` provide you an example to build imdb, Remember changing and adding new params.

- step3. Run `gen_12net_hard_example.py` to generate hard sample. Run `gen_24net_data.py`. Combine these output and generate imdb.

- step4. Similar to last step, Run `gen_24net_hard_example.py` to generate hard sample. Run `gen_48net_data.py`. Combine these output and generate imdb. 

Strongly suggest readers generate training data themselves. The sample training data of 12net and 24net is available( Too big for Baidu Drive to upload 48net) by sending [Email](cong_weilin@qq.com)
## Net
The main idea is block backward propagation for different task

12net
![12net](https://github.com/CongWeilin/mtcnn-caffe/blob/master/12net/train12.png)
24net
![24net](https://github.com/CongWeilin/mtcnn-caffe/blob/master/24net/train24.png)
48net
![48net](https://github.com/CongWeilin/mtcnn-caffe/blob/master/48net/train48.png)

## Questions
The Q&A bellow can solve most of your problem.

Q1: What data base do you use?<br/>
A1: Similar to official paper, Wider Face for detection and CelebA for alignment.

Q2: What is "12(24/48)net-only-cls.caffemodel" file for?<br/>
A2: Provide a initial weigh to train. Since caffe's initial weigh is random, a bad initial weigh may take a long ran to converge even might overfit before that.

Q3: Why preprocess images by minus 128?<br/>
A3: Separating data from (0,+) to (-,+), can make converge faster and more accurate. Refer to [Batch Normalization Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)

Q4: Do you implement OHEM(Online-Hard-Example-Mining)?<br/>
A4: No. OHEM is used when training data is not that much big. Refer to [faster-rcnn's writer RBG's paper](https://arxiv.org/pdf/1604.03540.pdf)

Q5: Ratio positive/negative samples for 12net?<br/>
A5: This caffemodel used neg:pos=3:1. Because 12net's function is to eliminate negative answers, similar to exclusive method, we should learn more about negative elininate the wrong answer.

Q6: Why your stride is different to official?<br/>
A6: If you input a (X,X) image, the output Y = (X-11)/2. Every point on output represent a ROI on input. The ROI's left side moving range = (0, X-12) on input, and (0, Y-1) on output. So that stride = (X-12)/(Y-1) ≈≈ 2 in this net.

Q7: What is roi(cls/pts).imdb used for?<br/>
A7: Use imdb can feed training data into training net faster. Instead of random search data from the hard-disk, reading data from a large file once to memory will save you a lot of time. `imdb` was created by python module-cPickle.

Q8: What is `tools_matrix.py` different from `tools.py`?<br/>
A8: Matrix version use linear matrix to make calculation faster(160ms on FDDB). If you are green hand in this area, read Non-Matrix version to understand each process.

Q9: I want your training data for emergency use. How to use them? How to train? ...<br/>
A9: ???
## Current Status
CongWeilin updated in 2017/3/5

Update `tools_matrix.py` to make calculate faster, about 160ms/image. 
