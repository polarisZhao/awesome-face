# awesome-face [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/polarisZhao/awesome-face)
😎  face releated algorithm, datasets and papers  😋

## 🎉 Paper / Algorithm

### 📌 Face Recognition 

#### 1. metric learning：

- **Deep ID Series**

​    **DeepID1:** [Deep Learning Face Representation from Predicting 10,000 Classes](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Sun_Deep_Learning_Face_2014_CVPR_paper.pdf) [Yi Sun et al., 2014]

​    **DeepID2:** [Deep Learning Face Representation by Joint Identification-Verification](https://arxiv.org/abs/1406.4773) [Yi Sun et al., 2014]

​    **DeepID2+:** [Deeply learned face representations are sparse, selective, and robust](https://arxiv.org/abs/1412.1265) [Yi Sun et al., 2014]

​    **DeepIDv3:** [DeepID3: Face Recognition with Very Deep Neural Networks](https://arxiv.org/abs/1502.00873) [Yi Sun et al., 2015]

- **FaceNet:**     [[**third-party implemention**]](https://github.com/davidsandberg/facenet)

​    [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832) [Florian Schroff et al., 2015]

- **Deep Face:**

​    [Deep Face Recognition](https://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf) [Omkar M. Parkhi et al., 2015]

#### 2. margin based classification

- **Center Loss：**   [[**code**]](https://github.com/ydwen/caffe-face)
  [A Discriminative Feature Learning Approach for Deep Face Recognition](http://ydwen.github.io/papers/WenECCV16.pdf) [Yandong Wen et al., 2016]
- **Large-Margin Softmax Loss**   [[**code**]](https://github.com/wy1iu/LargeMargin_Softmax_Loss)
  [Large-Margin Softmax Loss for Convolutional Neural Networks(L-Softmax loss)](https://arxiv.org/pdf/1612.02295.pdf) [Weiyang Liu al., 2017]

![center_norm](./img/softmax_center_l.png)

- **SphereFace：**  **A-Softmax**    [[**code**]](https://github.com/wy1iu/sphereface)
  [SphereFace: Deep Hypersphere Embedding for Face Recognition](https://arxiv.org/abs/1704.08063) [Weiyang Liu al., 2017]

- **NormFace**     [[**code**]](https://github.com/happynear/NormFace)

  [NormFace: L2 Hypersphere Embedding for Face Verification](https://arxiv.org/pdf/1704.06369.pdf) [Feng Wang al., 2017]

- **COCO Loss:**       [[**code**]](https://github.com/sciencefans/coco_loss)

  [Rethinking Feature Discrimination and Polymerization for Large-scale Recognition](https://arxiv.org/pdf/1710.00870.pdf) [Yu Liu al., 2017]

![A_L_COCO](./img/A_NORM_COCO.png)

- **AM-Softmax**   [[**code**]](https://github.com/happynear/AMSoftmax)
  [AM : Additive Margin Softmax for Face Verification](https://arxiv.org/pdf/1801.05599.pdf) [Feng Wang al., 2018]
- **CosFace:**

​    [CosFace: Large Margin Cosine Loss for Deep Face Recognition(Tencent AI Lab)](https://arxiv.org/pdf/1801.09414.pdf) [Hao Wang al., 2018]

- **ArcFace:**   [**[code]**](https://github.com/deepinsight/insightface )

​    [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/pdf/1801.07698.pdf) [Jiankang Deng al., 2018]

![cos_loss](./img/cos_loss.png)

- **CCL：**

  [Face Recognition via Centralized Coordinate Learning](https://arxiv.org/pdf/1801.05678.pdf) [Xianbiao al., 2018]


#### 3.  3D face recognition

- [Deep 3D Face Identification ](https://arxiv.org/abs/1703.10714)[Donghyun Kim al., 2017]

- [Learning from Millions of 3D Scans for Large-scale 3D Face Recognition](https://arxiv.org/abs/1711.05942)[Syed Zulqarnain al., 2018] 

#### 4. others

- [Beyond triplet loss: a deep quadruplet network for person re-identification](https://arxiv.org/pdf/1704.01719.pdf)[Weihua Chen al., 2017]
- [Range Loss for Deep Face Recognition with Long-tail](https://arxiv.org/abs/1611.08976) [Xiao Zhang al., 2016]
- [Feature Incay for Representation Regularization](https://arxiv.org/abs/1705.10284)[Yuhui Yuan al., 2017]

### 📌 Face Detection

- **Cascade**   [**[code]**](https://github.com/anson0910/CNN_face_detection)

  [A Convolutional Neural Network Cascade for Face Detection](https://ieeexplore.ieee.org/document/7299170/)[Haoxiang Li al., 2015]

- **MTCNN**    [**[code]**](https://github.com/kpzhang93/MTCNN_face_detection_alignment)

   [Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://kpzhang93.github.io/MTCNN_face_detection_alignment/)[K. Zhang al., 2016]

### 📌 Face Alignment

- [Look at Boundary: A Boundary-Aware Face Alignment Algorithm](https://arxiv.org/abs/1805.10483)[Wayne Wu al., 2018]

### 📌Others

- [Exploring Disentangled Feature Representation Beyond Face Identification](https://arxiv.org/abs/1804.03487v1)[Yu Liu al., 2018] 

## 📦  Datasets

### 📌 Face Recognition

| Datasets           | Description                                                  | Links                                                        |
| ------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **CASIA-WebFace**  | **10,575** subjects and **494,414** images                   | [Download](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html) |
| **MegaFace**🏅      | **1 million** faces, **690K** identities                     | [Download](http://megaface.cs.washington.edu/)               |
| **MS-Celeb-1M**🏅   | about **10M** images for **100K** celebrities   Concrete measurement to evaluate the performance of recognizing one million celebrities | [Download](http://www.msceleb.org)                           |
| **LFW**🏅           | **13,000** images of faces collected from the web. Each face has been labeled with the name of the person pictured.  **1680** of the people pictured have two or more distinct photos in the data set. | [Download](http://vis-www.cs.umass.edu/lfw/)                 |
| **VGG Face2**🏅     | The dataset contains **3.31 million** images of **9131** subjects (identities), with an average of 362.6 images for each subject. | [Download](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/)  |
| **YouTube Face**   | The data set contains **3,425** videos of **1,595** different people. | [Download](http://www.cs.tau.ac.il/%7Ewolf/ytfaces/)         |
| **IJB-B**          |                                                              | [Download](https://www.nist.gov/itl/iad/image-group/ijbb-dataset-request-form) |
| **FaceScrub**      | It comprises a total of **106,863** face images of male and female **530** celebrities, with about **200 images per person**. | [Download](http://vintage.winklerbros.net/facescrub.html)    |
| **Mut1ny**         | head/face segmentation dataset contains over 17.3k labeled images | [Download](http://www.mut1ny.com/face-headsegmentation-dataset) |
| **Trillion Pairs** | Train: **MS-Celeb-1M-v1c** &  **Asian-Celeb** Test: **ELFW & DELFW** | [Download](http://trillionpairs.deepglint.com/overview)      |

### 📌Face Detection

| Datasets        | Description                                                  | Links                                                        |
| --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **FDDB**🏅       | **5171** faces in a set of **2845** images                   | [Download](http://vis-www.cs.umass.edu/fddb/index.html)      |
| **Wider-face**🏅 | **32,203** images and label **393,703** faces with a high degree of variability in scale, pose and occlusion, organized based on **61** event classes | [Download](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/)  |
| **AFW**         | AFW dataset is built using Flickr images. It has **205** images with **473** labeled faces. For each face, annotations include a rectangular **bounding box**, **6 landmarks** and the **pose angles**. | [Download](http://www.ics.uci.edu/~xzhu/face/)               |
| **MALF**        | MALF is the first face detection dataset that supports fine-gained evaluation. MALF consists of **5,250** images and **11,931** faces. | [Download](http://www.cbsr.ia.ac.cn/faceevaluation/)         |
| **IJB-A**       | IJB-A is proposed for face detection and face recognition. IJB-A contains **24,327** images and **49,759** faces. | [Download](https://www.nist.gov/itl/iad/image-group/ijb-dataset-request-form) |

### 📌 Face Attributes & Keypoints

| Datasets                 | Description                                                  | Links                                                        |
| ------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **CelebA**               | **10,177** number of **identities**,  **202,599** number of **face images**, and  **5 landmark locations**, **40 binary attributes** annotations per image. | [Download](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) |
| **IMDB-WIKI**            | 500k+ face images with **age** and **gender** labels         | [Download](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) |
| **Adience**              | Unfiltered faces for **gender** and **age** classification   | [Download](http://www.openu.ac.il/home/hassner/Adience/data.html) |
| **CACD2000**             | The dataset contains more than 160,000 images of 2,000 celebrities with **age ranging from 16 to 62**. | [Download](http://bcsiriuschen.github.io/CARC/)              |
| **Caltech10k Web Faces** | The dataset has 10,524 human faces of various resolutions and in **different settings** | [Download](http://www.vision.caltech.edu/Image_Datasets/Caltech_10K_WebFaces/#Description) |
| **WFLW**                 | WFLW contains **10000 faces** (7500 for training and 2500 for testing) with **98 fully manual annotated landmarks**. | [Download](https://wywu.github.io/projects/LAB/WFLW.html)    |

### 📌 Others：

| Datasets                   | Description                                                  | Links                                                        |
| -------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **3D Mask Attack** Dataset | **76500** frames of **17** persons using Kinect RGBD with eye positions (Sebastien Marcel). | [Download](https://www.idiap.ch/dataset/3dmad)               |
| **MOBIO**                  | **bi-modal** (**audio** and **video**) data taken from 152 people. | [Download](https://www.idiap.ch/dataset/mobio)               |
| **BANCA**                  | The BANCA database was captured in four European languages in **two modalities** (**face** and **voice**). | [Download](http://www.ee.surrey.ac.uk/CVSSP/banca/)          |
| **B3D(AC)^2**              | **1000** high quality, dynamic **3D scans** of faces, recorded while pronouncing a set of English sentences. | [Download](http://www.vision.ee.ethz.ch/datasets/b3dac2.en.html) |
| **BD-3DFE**                | Analyzing **Facial Expressions** in **3D** Space             | [Download](http://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html) |
| **Bosphorus**              | 105 subjects and 4666 faces 2D & 3D face data                | [Download](http://bosphorus.ee.boun.edu.tr/default.aspx)     |
| **ND-2006**                | 422 subjects and 9443 faces 3D Face Recognition              | [Download](https://sites.google.com/a/nd.edu/public-cvrl/data-sets) |
| **FRGC V2.0**              | 466 subjects and 4007 of 3D Face, Visible Face Images        | [Download](https://sites.google.com/a/nd.edu/public-cvrl/data-sets) |

## 🔧 References:

[1] <https://github.com/RiweiChen/DeepFace/tree/master/FaceDataset>

[2] <https://www.zhihu.com/question/33505655?sort=created>

[3] https://github.com/betars/Face-Resources

[4] https://zhuanlan.zhihu.com/p/33288325

[5] https://github.com/L706077/DNN-Face-Recognition-Papers

[6] https://www.zhihu.com/question/67919300

## 🕳 ToDo

- [ ] Software
- [ ] add video、few sample and Disguised face recognition algorithm

