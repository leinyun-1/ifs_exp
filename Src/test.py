def test_HG():
    import torch 
    from Model.HourGlass import HGFilter
    from types import SimpleNamespace
    config = {
        "num_stack": 4,
        "num_hourglass": 2,
        "hourglass_dim": 256,
        "hg_down": 'ave_pool',
        "norm": 'group'
    }
    config = SimpleNamespace(**config)
    net = HGFilter(config).cuda()
    x = torch.randn(1,3,512,512).cuda()
    y = net(x)
    print(y.shape)

def test_mlp():
    import torch 
    from Model.SurfaceClassifier import SurfaceClassifier
    chnnels = [323, 1024, 512, 256, 128, 1]
    net = SurfaceClassifier(chnnels).cuda()
    x = torch.randn(1,323,8000).cuda()
    y = net(x)
    print(y.shape)

def make_train_txt():
    file_path = "/root/leinyu/data/thuman2/ft_local/dataset/Res2048/0418/train.txt"
    with open(file_path,'w') as f:
        for i in range(421):
            f.write(f"{i:04d}\n")

def test_pifu():
    import torch 
    from Model.Pifu import IFSNet
    from Dataset.PifuIFS import get_dataloader,ifs_pack

    net = IFSNet(tm=None,rov=-1).cuda()
    dataset = get_dataloader(path="/root/leinyu/data/thuman2/ft_local/dataset ../grid_samples_64_12")
    dataloader = torch.utils.data.DataLoader(dataset)
    for b_data in dataloader:
        b_data = ifs_pack('cuda',b_data)
        print(b_data['images'].shape)
        y,loss,_ = net.forward(epoch=0,bidx=0,data=b_data)
        print(y.shape)
        print(loss)
        break 

def make_dataset():
    import os 
    import shutil
    import pickle 
    from PIL import Image 
    dataset_root = '/root/leinyu/data/thuman2/ft_local/dataset/Res2048/0418'
    image_root = os.path.join(dataset_root,'image')
    mask_root = os.path.join(dataset_root,'mask')
    param_path = os.path.join(dataset_root,'ProjParams.pkl')

    eval_image_root = '/mnt/aigc_cq/private/leinyu/code/skyreels_v2/result/i2v_1.3b_lora_0822/0825/images'
    eval_images = sorted(os.listdir(eval_image_root))
    eval_images = eval_images[::10][:-1] # 从0-80共81图取出均匀环绕8图
    for i,image in enumerate(eval_images):
        src_path = os.path.join(eval_image_root,image)
        dest_path = os.path.join(image_root,'woman_'+ str(i*6)+'.png')
        dest_mask_path = os.path.join(mask_root,'woman_'+str(i*6)+'_mask.png')

        Image.open(src_path).resize((2048,2048), resample=Image.NEAREST).save(dest_path)
        Image.open(src_path).resize((2048,2048), resample=Image.NEAREST).save(dest_mask_path)
        #shutil.copy(src_path,dest_path)

    # cam_params = pickle.load(open(param_path, "rb"), encoding="iso-8859-1")

    # views = [0, 6, 12, 18, 24, 30, 36, 42]
    # for vid in views:
    #     param = {}
    #     param['K'] = cam_params['0000_'+str(vid)]['K']
    #     param['R'] = cam_params['0000_'+str(vid)]['R']
    #     param['t'] = cam_params['0000_'+str(vid)]['t']
    #     cam_params['woman_' + str(vid)] = param.copy()
    
    # # 回存更新后的相机参数到原始路径
    # with open(param_path, 'wb') as f:
    #     pickle.dump(cam_params, f)
    
    





if __name__ == "__main__":
    #test_HG()
    #test_mlp()
    #make_train_txt()
    #test_pifu()
    make_dataset()