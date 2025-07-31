# 命名
* volume，空间体，指整个空间
* voxel，体素
* sample，采样点
*
* image，原render，图像
* patch，特指crop之后的图像块
*
## 部分缩写
* 原则上常用缩写应加入列表，若不在列表中的，须在程序中注释
* 有实体的是数量，无实体的是粒度

* 分辨率、粒度
  * rov，原local_res，resolution of voxel，指一个voxel的分辨率
  * rom，原resolution，resolution of volume，指一个volume的分辨率，应等于rov * som
  * som，原center_res，size of volume，指volume三个轴三划分的voxel数量
  * roi，原load_size或loadSize，resolution of image，使用的图像分辨率

* 数量
  * nov，number of views，视角数
  * vpv，voxels per volume，这是一个volume中voxel的数量，等于som^3
  * spv，原res，samples per voxel，一个voxel里的samples数量，等于rov^3

结构
* 不集中、分散：option，yaml，代码中，命令行参数
* 太依赖opt，不必要的不用配置，开关意味着可能性，可能性越多意味着边界越模糊，必要时甚至用多类分别对待各种配置都可以
  * 去除local、projection、vh_method、error_term、use_ml_feat等算法高内聚的开关
* 各版本的参数存在差异，缺乏一致性，例如mc_threshold
* 代码结构交错，层次结构不清晰
* eval部分和net耦合性太强，不该涉及太多细节
* 去除当前网络版本无用的函数
  * mask_init、norm_init、smpl_init
  * index_3d、index3D
* dataset专有化，配合网络版本，不必大而全
* 两种数据集读取方式应该分离

实现
* o SurfaceClassifier无用去除
  * mlp_dim无用，去除
* o perspective中transforms没用，移除，extrinstic没用，移除
* frozen用detach
* conv_for_l移入unet，变成outputlayer
  * unet变一个专属名字，移入ifsnet，组件保持共用，保留在net_utils
* filter2d输出变为局部变量，由此，query时需传入，这个流程也能在之后继续使能维持feature固定的情况下query，且流程更具可读性
* 算法中数据在numpy、torch中来回切换，影响效率
* forward仅分成两部分，编码+query，不要把index也独立出来，而query等时可以外部用的，其他更底层的不再对外
  * 非必要——例如效率考量，尽量用forward，而不分别用filter2d+query
* num_views动态化：由数据决定而不需要固化网络，同一次处理样本间可不一致
* patch_legth = int(self.patch_length[0])这个把B和patch_length放一起的实现不够优雅
  * 去除了，没必要有，直接计算就好
* crop的size是硬算的，算法和voxel本身无关
  * patch_img_size = self.load_size // self.center_res * 2
  * 就算要考虑对齐，至少统计下，bbox的大小，和patch_img_size的关系，注意要考虑感受野
* 重做crop算法，包括mask_crop的
* 用成员变量存储中间结果不合理，职责不符，应变为局部变量/参数
* 函数命名可读性低
  * vh, filter_2d等
* 变量命名可读性低
  * xyz等



算法级别
* query的并行度要考量下，看看是不是有问题
  * feature并行度
    * unet并行度
    * conv_for_l并行度
  * index并行度
  * 反投影并行度
  * query并行度
* projection 改成 B、V批量的形式计算，calibration本身是按照B、V堆叠的
* crop没有居中
* 目前view尺寸不一致，之后视情况一致后去除
* 两次做sdf，应该是因为考虑local的边界问题，这个不该这样，应该是直接做全局的
* 不该有patch_legth的概念，应直接变成大的patch
* 采样应该重做
  * 第一、第二部分重复
  * 第四部分应用均匀采样，不应该还是用高斯


# 问题
* 为什么patch不居中？@ PatchedIFS
  * 而要这么做？bbox[0] = (bbox[0] // factor) * factor
* mask_crop的mask是怎么算的，有没有留余量？
* center和sample是否计算错？
  * centers, _ = create_grid(som, som, som, b_min + l / (rom - 1) * (rov - 1) / 2, b_max - l / (rom - 1) * (rov - 1) / 2)
  * create_grid(rom, rom, rom, b_min, b_max)
  * b_min = b_min + l / (rom - 1) * (rov // 2)
* mask_crop的bbox是怎么算的
* np.concatenate([centers_near, centers_near, centers_near_1, empty_centers], axis=0)
  * todo: 为什么不干脆多一倍数量num_centers * 2
* 数据集总数为200，但只用了前100，到底是哪部分
* 没有背景，黑色变成一种mask




