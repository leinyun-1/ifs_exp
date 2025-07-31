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

if __name__ == "__main__":
    test_HG()
    #test_mlp()
    #make_train_txt()
    #test_pifu()