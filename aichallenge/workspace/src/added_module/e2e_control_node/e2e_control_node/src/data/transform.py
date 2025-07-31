import torchvision.transforms as transforms

class DictTransform:
    """
    辞書形式のサンプルを受け取り、'image'キーの値にのみ画像変換を適用するクラス
    """
    def __init__(self, image_transform):
        self.image_transform = image_transform

    def __call__(self, sample):
        #NumPy配列(H, W, C)を想定
        sample['image'] = self.image_transform(sample['image'])
        return sample

def get_transforms():
    """
    学習用(train)と検証用(val)のTransformを辞書形式で返す関数
    """
    # 学習用の画像Transform (データ拡張を含む)
    train_image_transforms = transforms.Compose([
        transforms.ToTensor(), # NumPy(H,W,C) -> Tensor(C,H,W) & [0, 255] -> [0.0, 1.0]
        transforms.Resize((224, 224), antialias=True),
        transforms.RandomHorizontalFlip(p=0.5), # 50%の確率で左右反転
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # 明るさとコントラストをランダムに変更
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 検証・テスト用の画像Transform (データ拡張を含まない)
    val_image_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transforms_dict = {
        'train': DictTransform(image_transform=train_image_transforms),
        'val': DictTransform(image_transform=val_image_transforms)
    }

    return transforms_dict