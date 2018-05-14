# awesome-face
face releated algorithm, datasets and papers

### Datasets

​    本页面主要汇总了常见的人脸检测，人脸识别，人脸属性(年龄，性别，表情，关键点等)的相关数据集。

人脸识别

| 数据库            | 描述                                                     | 用途                 | 获取方法                                                     |
| ----------------- | -------------------------------------------------------- | -------------------- | ------------------------------------------------------------ |
| **CASIA-WebFace** | 10k+人，约500K张图片                                     | 非限制场景           | [链接](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html) |
| **FaceScrub**     | 530人，约100k张图片                                      | 非限制场景           | [链接](http://vintage.winklerbros.net/facescrub.html)﻿(失效)  |
| YouTube Face      | 1,595个人 3,425段视频                                    | 非限制场景、视频     | [链接](http://www.cs.tau.ac.il/%7Ewolf/ytfaces/)             |
| LFW               | 5k+人脸，超过10K张图片                                   | 标准的人脸识别数据集 | [链接](http://vis-www.cs.umass.edu/lfw/)                     |
| MultiPIE          | 337个人的不同姿态、表情、光照的人脸图像，共750k+人脸图像 | 限制场景人脸识别     | [链接](http://www.multipie.org/) 需购买                      |
| MegaFace          | 690k不同的人的1M人脸图像                                 | 新的人脸识别评测集合 | [链接](http://megaface.cs.washington.edu/)                   |
| IJB-A             |                                                          | 人脸识别，人脸检测   | [链接](http://www.nist.gov/itl/iad/ig/ijba_request.cfm)      |
| CAS-PEAL          | 1040个人的30k+张人脸图像，主要包含姿态、表情、光照变化   | 限制场景下人脸识别   | [链接](http://www.jdl.ac.cn/peal/index.html)                 |
| **MS-Celeb-1M**   | 100k个人的1M+人脸图像                                    | 微软人脸识别数据集   | [链接](http://www.msceleb.org)                               |

Face Detection

| 数据库               | 描述                               | 用途               | 获取方法                                                     |
| -------------------- | ---------------------------------- | ------------------ | ------------------------------------------------------------ |
| FDDB                 | 2845张图片中的5171张脸             | 标准人脸检测评测集 | [链接](http://vis-www.cs.umass.edu/fddb/)                    |
| IJB-A                |                                    | 人脸识别，人脸检测 | [链接](http://www.nist.gov/itl/iad/ig/ijba_request.cfm)      |
| Caltech10k Web Faces | 10k+人脸，提供双眼和嘴巴的坐标位置 | 人脸点检测         | [链接](http://www.vision.caltech.edu/Image_Datasets/Caltech_10K_WebFaces/#Description) |
| **Wider-face**       |                                    | 人脸检测           |                                                              |

人脸年龄和性别属性

| 数据库    | 描述                                                         | 用途                      | 获取方法                                                     |
| --------- | ------------------------------------------------------------ | ------------------------- | ------------------------------------------------------------ |
| IMDB-WIKI | 包含：IMDb中20k+个名人的460k+张图片 和维基百科62k+张图片, 总共： 523k+张图片 | 名人年龄、性别            | [链接](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) |
| Adience   | 包含2k+个人的26k+张人脸图像                                  | 人脸性别，人脸年龄段(8组) | [链接](http://www.openu.ac.il/home/hassner/Adience/data.html) |
| CACD2000  | 2k名人160k张人脸图片(跨年龄)                                 | 人脸年龄                  | [链接](http://bcsiriuschen.github.io/CARC/)                  |

人脸属性识别表

| 数据库     | 描述                         | 用途         | 获取方法                                                 |
| ---------- | ---------------------------- | ------------ | -------------------------------------------------------- |
| **CelebA** | 200k张人脸图像40多种人脸属性 | 人脸属性识别 | [链接](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) |

参考资料：

1. <https://github.com/RiweiChen/DeepFace/tree/master/FaceDataset>

2. <https://www.zhihu.com/question/33505655?sort=created>