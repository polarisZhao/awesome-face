# awesome-face [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/polarisZhao/awesome-face)
😎  face releated algorithm, datasets and papers  😋

### 🎉 Papers

##### Metrci Learning：

**Deep ID Series**

**DeepID1:**[Deep Learning Face Representation from Predicting 10,000 Classes](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Sun_Deep_Learning_Face_2014_CVPR_paper.pdf) [Yi Sun et al., 2014]

**DeepID2:**[Deep Learning Face Representation by Joint Identification-Verification](https://arxiv.org/abs/1406.4773) [Yi Sun et al., 2014]

**DeepID2+:**[Deeply learned face representations are sparse, selective, and robust](https://arxiv.org/abs/1412.1265) [Yi Sun et al., 2014]

**DeepIDv3:**[DeepID3: Face Recognition with Very Deep Neural Networks](https://arxiv.org/abs/1502.00873) [Yi Sun et al., 2015]

**FaceNet:**

[FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832) [Florian Schroff et al., 2015]

**Deep Face:**

[Deep Face Recognition](https://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf) [Omkar M. Parkhi et al., 2015]

##### Margin Based Classification

**Center Loss：**[A Discriminative Feature Learning Approach for Deep Face Recognition](http://ydwen.github.io/papers/WenECCV16.pdf) [Yandong Wen et al., 2016]

[Large-Margin Softmax Loss for Convolutional Neural Networks(L-Softmax loss)](https://arxiv.org/pdf/1612.02295.pdf) [Weiyang Liu al., 2017]

[SphereFace: Deep Hypersphere Embedding for Face Recognition](https://arxiv.org/abs/1704.08063) [Weiyang Liu al., 2017]

[NormFace: L2 Hypersphere Embedding for Face Verification](https://arxiv.org/pdf/1704.06369.pdf) [Feng Wang al., 2017]

[AM : Additive Margin Softmax for Face Verification](https://arxiv.org/pdf/1801.05599.pdf) [Feng Wang al., 2018]

[AAM : Face Recognition via Centralized Coordinate Learning](https://arxiv.org/pdf/1801.05678.pdf) [Xianbiao al., 2018]

[ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/pdf/1801.07698.pdf) [Jiankang Deng al., 2018]

[CosFace: Large Margin Cosine Loss for Deep Face Recognition(Tencent AI Lab)](https://arxiv.org/pdf/1801.09414.pdf) [Hao Wang al., 2018]

**Other loss function：**

[Beyond triplet loss: a deep quadruplet network for person re-identification](https://arxiv.org/pdf/1704.01719.pdf) [Weihua Chen et al., 2017]

[Range Loss for Deep Face Recognition with Long-tail](https://arxiv.org/abs/1611.08976) [Xiao Zhang et al., 2016]

[L2-constrained Softmax Loss for Discriminative Face Verification](https://arxiv.org/abs/1703.09507v2) [Rajeev Ranjan al., 2017]

[Rethinking Feature Discrimination and Polymerization for Large-scale Recognition(CoCo loss)](https://arxiv.org/pdf/1710.00870.pdf) [Yu Liu al., 2017]

[DeepVisage: Making face recognition simple yet with powerful generalization skills](https://arxiv.org/abs/1703.08388) [Abul Hasnat al., 2017]

##### Pruning Network

[DSD: Dense-Sparse-Dense Training for Deep Neural Networks](https://arxiv.org/abs/1607.04381) [Song Han et al., 2017]

[Pruning Convolutional Neural Networks for Resource Efficient Transfer Learning](https://arxiv.org/abs/1611.06440) [Pavlo Molchanov et al., 2017]

[Learning both Weights and Connections for Efficient Neural Networks](https://arxiv.org/abs/1506.02626) [Song Han et al., 2016]

##### 3D Face Recognition：

[Deep 3D Face Identification](https://arxiv.org/pdf/1703.10714.pdf) [Donghyun Kim al., 2017]

[Learning from Millions of 3D Scans for Large-scale 3D Face Recognition](https://arxiv.org/pdf/1711.05942.pdf) [S. Z. Gilani al.,2018]



### 📦  Datasets

#### Face Recognition

| Datasets          | Description                                                  | Links                                                        |
| ----------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **CASIA-WebFace** | **10,575** subjects and **494,414** images                   | [Download](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html) |
| **MegaFace**      | **1 million** faces, **690K** identities                     | [Download](http://megaface.cs.washington.edu/)               |
| **MS-Celeb-1M**   | about **10M** images for **100K** celebrities   Concrete measurement to evaluate the performance of recognizing one million celebrities | [Download](http://www.msceleb.org)                           |
| **LFW**           | **13,000** images of faces collected from the web. Each face has been labeled with the name of the person pictured.  **1680** of the people pictured have two or more distinct photos in the data set. | [Download](http://vis-www.cs.umass.edu/lfw/)                 |
| **VGG Face2**     | The dataset contains **3.31 million** images of **9131** subjects (identities), with an average of 362.6 images for each subject. | [Download](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/)  |
| **YouTube Face**  | The data set contains **3,425** videos of **1,595** different people. | [Download](http://www.cs.tau.ac.il/%7Ewolf/ytfaces/)         |
| **IJB-B**         |                                                              | [Download](https://www.nist.gov/itl/iad/image-group/ijbb-dataset-request-form) |
| **FaceScrub**     | It comprises a total of **106,863** face images of male and female **530** celebrities, with about **200 images per person**. | [Download](http://vintage.winklerbros.net/facescrub.html)    |

#### Face Detection

| Datasets       | Description                                                  | Links                                                        |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **FDDB**       | **5171** faces in a set of **2845** images                   | [Download](http://vis-www.cs.umass.edu/fddb/index.html)      |
| **Wider-face** | **32,203** images and label **393,703** faces with a high degree of variability in scale, pose and occlusion, organized based on **61** event classes | [Download](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/)  |
| **AFW**        | AFW dataset is built using Flickr images. It has **205** images with **473** labeled faces. For each face, annotations include a rectangular **bounding box**, **6 landmarks** and the **pose angles**. | [Download](http://www.ics.uci.edu/~xzhu/face/)               |
| **MALF**       | MALF is the first face detection dataset that supports fine-gained evaluation. MALF consists of **5,250** images and **11,931** faces. | [Download](http://www.cbsr.ia.ac.cn/faceevaluation/)         |
| **IJB-A**      | IJB-A is proposed for face detection and face recognition. IJB-A contains **24,327** images and **49,759** faces. | [Download](https://www.nist.gov/itl/iad/image-group/ijb-dataset-request-form) |

#### Face Attributes & Keypoints

| Datasets                 | Description                                                  | Links                                                        |
| ------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **CelebA**               | **10,177** number of **identities**,  **202,599** number of **face images**, and  **5 landmark locations**, **40 binary attributes** annotations per image. | [Download](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) |
| **IMDB-WIKI**            | 500k+ face images with **age** and **gender** labels         | [Download](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) |
| **Adience**              | Unfiltered faces for **gender** and **age** classification   | [Download](http://www.openu.ac.il/home/hassner/Adience/data.html) |
| **CACD2000**             | The dataset contains more than 160,000 images of 2,000 celebrities with **age ranging from 16 to 62**. | [Download](http://bcsiriuschen.github.io/CARC/)              |
| **Caltech10k Web Faces** | The dataset has 10,524 human faces of various resolutions and in **different settings** | [Download](http://www.vision.caltech.edu/Image_Datasets/Caltech_10K_WebFaces/#Description) |

#### Others：

| Datasets                                                     | Description                                                  | Links                                                        |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **3D Mask Attack** Dataset                                   | **76500** frames of **17** persons using Kinect RGBD with eye positions (Sebastien Marcel). | [Download](https://www.idiap.ch/dataset/3dmad)               |
| **MOBIO**                                                    | **bi-modal** (**audio** and **video**) data taken from 152 people. | [Download](https://www.idiap.ch/dataset/mobio)               |
| **BANCA**                                                    | The BANCA database was captured in four European languages in **two modalities** (**face** and **voice**). | [Download](http://www.ee.surrey.ac.uk/CVSSP/banca/)          |
| **Biwi 3D Audiovisual Corpus of Affective Communication**    | **1000** high quality, dynamic **3D scans** of faces, recorded while pronouncing a set of English sentences. | [Download](http://www.vision.ee.ethz.ch/datasets/b3dac2.en.html) |
| **Binghampton Univ 3D static and dynamic facial expression** | Analyzing **Facial Expressions** in **3D** Space             | [Download](http://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html) |

#### 🔧 References:

1. <https://github.com/RiweiChen/DeepFace/tree/master/FaceDataset>
2. <https://www.zhihu.com/question/33505655?sort=created>
3. https://github.com/betars/Face-Resources
4. https://zhuanlan.zhihu.com/p/33288325
5. https://github.com/L706077/DNN-Face-Recognition-Papers



#### 🕳 ToDo

- [ ] add papers releated code
- [ ] mark common beachmark