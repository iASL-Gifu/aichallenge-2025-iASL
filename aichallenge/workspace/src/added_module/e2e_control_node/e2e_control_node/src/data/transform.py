from torchvision import transforms

IMG_SIZE = (224, 224)
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

# 学習用transformパイプライン
train_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(), # PIL Imageを[0, 1]のTensorに変換
    transforms.Normalize(mean=IMG_MEAN, std=IMG_STD) # Tensorを正規化
])

# 検証/テスト用transformパイプライン (データ拡張は行わない)
val_test_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMG_MEAN, std=IMG_STD)
])

# transformをかけない場合（Tensorに変換するだけなど）も定義しておくと便利
minimal_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
])