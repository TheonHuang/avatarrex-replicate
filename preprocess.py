from torchvision import transforms

# 定义预处理流水线
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整图像大小
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
])

# 对输入数据进行预处理
input_data = preprocess(input_data)
