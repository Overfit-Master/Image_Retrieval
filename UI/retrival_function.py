import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
import Retrival.model as Model
import torchvision.transforms as transforms
import PIL.Image as Image
import numpy as np
import sqlite3
import io

# 网络模型
flower_model = Model.flower_model
car_model = Model.car_model
classify_model = Model.classify_model

# 权重文件路径
flower_weight_path = r""
car_weight_path = r""
classify_weight_path = r""

# 对输入图像进行处理再传入
transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 数据库路径
db_path = r""


# 先对输入图像进行分类，判断是车还是花
def classify_input(image):
    classify_dict = {"flower": 0, "car": 1}
    classify_model.load_state_dict(torch.load(classify_weight_path))
    classify_model.to("cuda:0").eval()
    image_tensor = transform(image).to("cuda:0").unsqueeze(0)
    with torch.no_grad():
        output = classify_model(image_tensor)
        predict = output.argmax(dim=1).item()
    # 根据值查找对应键
    for key, val in classify_dict.items():
        if val == predict:
            return key


# 计算传入系统图像的哈希码
def calculate_feature_code(image):
    flower_dict = {"Bluebell": 0, "Buttercup": 1, "Colts'Foot": 2, "Cowslip": 3, "Crocus": 4, "Daffodil": 5, "Daisy": 6,
                   "Dandelion": 7, "Fritillary": 8, "Iris": 9, "LilyValley": 10, "Pansy": 11, "Snowdrop": 12,
                   "Sunflower": 13, "Tigerlily": 14, "Tulip": 15, "Windflower": 16}
    car_dict = {"Acura Integra Type R 2001": 0, "Acura RL Sedan 2012": 1, "Acura TL Sedan 2012": 2,
                "Acura TL Type-S 2008": 3, "Acura TSX Sedan 2012": 4, "Acura ZDX Hatchback 2012": 5,
                "AM General Hummer SUV 2000": 6, "Aston Martin V8 Vantage Convertible 2012": 7,
                "Aston Martin V8 Vantage Coupe 2012": 8, "Aston Martin Virage Convertible 2012": 9,
                "Aston Martin Virage Coupe 2012": 10, "Audi 100 Sedan 1994": 11, "Audi 100 Wagon 1994": 12,
                "Audi A5 Coupe 2012": 13, "Audi R8 Coupe 2012": 14, "Audi RS 4 Convertible 2008": 15,
                "Audi S4 Sedan 2007": 16, "Audi S4 Sedan 2012": 17, "Audi S5 Convertible 2012": 18,
                "Audi S5 Coupe 2012": 19, "Audi S6 Sedan 2011": 20, "Audi TT Hatchback 2011": 21,
                "Audi TT RS Coupe 2012": 22, "Audi TTS Coupe 2012": 23, "Audi V8 Sedan 1994": 24,
                "Bentley Arnage Sedan 2009": 25, "Bentley Continental Flying Spur Sedan 2007": 26,
                "Bentley Continental GT Coupe 2007": 27, "Bentley Continental GT Coupe 2012": 28,
                "Bentley Continental Supersports Conv. Convertible 2012": 29, "Bentley Mulsanne Sedan 2011": 30,
                "BMW 1 Series Convertible 2012": 31, "BMW 1 Series Coupe 2012": 32, "BMW 3 Series Sedan 2012": 33,
                "BMW 3 Series Wagon 2012": 34, "BMW 6 Series Convertible 2007": 35,
                "BMW ActiveHybrid 5 Sedan 2012": 36, "BMW M3 Coupe 2012": 37, "BMW M5 Sedan 2010": 38,
                "BMW M6 Convertible 2010": 39, "BMW X3 SUV 2012": 40, "BMW X5 SUV 2007": 41, "BMW X6 SUV 2012": 42,
                "BMW Z4 Convertible 2012": 43, "Bugatti Veyron 16.4 Convertible 2009": 44,
                "Bugatti Veyron 16.4 Coupe 2009": 45, "Buick Enclave SUV 2012": 46, "Buick Rainier SUV 2007": 47,
                "Buick Regal GS 2012": 48, "Buick Verano Sedan 2012": 49, "Cadillac CTS-V Sedan 2012": 50,
                "Cadillac Escalade EXT Crew Cab 2007": 51, "Cadillac SRX SUV 2012": 52,
                "Chevrolet Avalanche Crew Cab 2012": 53, "Chevrolet Camaro Convertible 2012": 54,
                "Chevrolet Cobalt SS 2010": 55, "Chevrolet Corvette Convertible 2012": 56,
                "Chevrolet Corvette Ron Fellows Edition Z06 2007": 57, "Chevrolet Corvette ZR1 2012": 58,
                "Chevrolet Express Cargo Van 2007": 59, "Chevrolet Express Van 2007": 60, "Chevrolet HHR SS 2010": 61,
                "Chevrolet Impala Sedan 2007": 62, "Chevrolet Malibu Hybrid Sedan 2010": 63,
                "Chevrolet Malibu Sedan 2007": 64, "Chevrolet Monte Carlo Coupe 2007": 65,
                "Chevrolet Silverado 1500 Classic Extended Cab 2007": 66,
                "Chevrolet Silverado 1500 Extended Cab 2012": 67, "Chevrolet Silverado 1500 Hybrid Crew Cab 2012": 68,
                "Chevrolet Silverado 1500 Regular Cab 2012": 69, "Chevrolet Silverado 2500HD Regular Cab 2012": 70,
                "Chevrolet Sonic Sedan 2012": 71, "Chevrolet Tahoe Hybrid SUV 2012": 72,
                "Chevrolet TrailBlazer SS 2009": 73, "Chevrolet Traverse SUV 2012": 74, "Chrysler 300 SRT-8 2010": 75,
                "Chrysler Aspen SUV 2009": 76, "Chrysler Crossfire Convertible 2008": 77,
                "Chrysler PT Cruiser Convertible 2008": 78, "Chrysler Sebring Convertible 2010": 79,
                "Chrysler Town and Country Minivan 2012": 80, "Daewoo Nubira Wagon 2002": 81,
                "Dodge Caliber Wagon 2007": 82, "Dodge Caliber Wagon 2012": 83, "Dodge Caravan Minivan 1997": 84,
                "Dodge Challenger SRT8 2011": 85, "Dodge Charger Sedan 2012": 86, "Dodge Charger SRT-8 2009": 87,
                "Dodge Dakota Club Cab 2007": 88, "Dodge Dakota Crew Cab 2010": 89, "Dodge Durango SUV 2007": 90,
                "Dodge Durango SUV 2012": 91, "Dodge Journey SUV 2012": 92, "Dodge Magnum Wagon 2008": 93,
                "Dodge Ram Pickup 3500 Crew Cab 2010": 94, "Dodge Ram Pickup 3500 Quad Cab 2009": 95,
                "Dodge Sprinter Cargo Van 2009": 96, "Eagle Talon Hatchback 1998": 97,
                "Ferrari 458 Italia Convertible 2012": 98, "Ferrari 458 Italia Coupe 2012": 99,
                "Ferrari California Convertible 2012": 100, "Ferrari FF Coupe 2012": 101, "FIAT 500 Abarth 2012": 102,
                "FIAT 500 Convertible 2012": 103, "Fisker Karma Sedan 2012": 104, "Ford E-Series Wagon Van 2012": 105,
                "Ford Edge SUV 2012": 106, "Ford Expedition EL SUV 2009": 107, "Ford F-150 Regular Cab 2007": 108,
                "Ford F-150 Regular Cab 2012": 109, "Ford F-450 Super Duty Crew Cab 2012": 110,
                "Ford Fiesta Sedan 2012": 111, "Ford Focus Sedan 2007": 112, "Ford Freestar Minivan 2007": 113,
                "Ford GT Coupe 2006": 114, "Ford Mustang Convertible 2007": 115, "Ford Ranger SuperCab 2011": 116,
                "Geo Metro Convertible 1993": 117, "GMC Acadia SUV 2012": 118, "GMC Canyon Extended Cab 2012": 119,
                "GMC Savana Van 2012": 120, "GMC Terrain SUV 2012": 121, "GMC Yukon Hybrid SUV 2012": 122,
                "Honda Accord Coupe 2012": 123, "Honda Accord Sedan 2012": 124, "Honda Odyssey Minivan 2007": 125,
                "Honda Odyssey Minivan 2012": 126, "HUMMER H2 SUT Crew Cab 2009": 127, "HUMMER H3T Crew Cab 2010": 128,
                "Hyundai Accent Sedan 2012": 129, "Hyundai Azera Sedan 2012": 130, "Hyundai Elantra Sedan 2007": 131,
                "Hyundai Elantra Touring Hatchback 2012": 132, "Hyundai Genesis Sedan 2012": 133,
                "Hyundai Santa Fe SUV 2012": 134, "Hyundai Sonata Hybrid Sedan 2012": 135,
                "Hyundai Sonata Sedan 2012": 136, "Hyundai Tucson SUV 2012": 137,
                "Hyundai Veloster Hatchback 2012": 138, "Hyundai Veracruz SUV 2012": 139,
                "Infiniti G Coupe IPL 2012": 140, "Infiniti QX56 SUV 2011": 141, "Isuzu Ascender SUV 2008": 142,
                "Jaguar XK XKR 2012": 143, "Jeep Compass SUV 2012": 144, "Jeep Grand Cherokee SUV 2012": 145,
                "Jeep Liberty SUV 2012": 146, "Jeep Patriot SUV 2012": 147, "Jeep Wrangler SUV 2012": 148,
                "Lamborghini Aventador Coupe 2012": 149, "Lamborghini Diablo Coupe 2001": 150,
                "Lamborghini Gallardo LP 570-4 Superleggera 2012": 151, "Lamborghini Reventon Coupe 2008": 152,
                "Land Rover LR2 SUV 2012": 153, "Land Rover Range Rover SUV 2012": 154,
                "Lincoln Town Car Sedan 2011": 155, "Maybach Landaulet Convertible 2012": 156,
                "Mazda Tribute SUV 2011": 157, "McLaren MP4-12C Coupe 2012": 158,
                "Mercedes-Benz 300-Class Convertible 1993": 159, "Mercedes-Benz C-Class Sedan 2012": 160,
                "Mercedes-Benz E-Class Sedan 2012": 161, "Mercedes-Benz S-Class Sedan 2012": 162,
                "Mercedes-Benz SL-Class Coupe 2009": 163, "Mercedes-Benz Sprinter Van 2012": 164,
                "MINI Cooper Roadster Convertible 2012": 165, "Mitsubishi Lancer Sedan 2012": 166,
                "Nissan 240SX Coupe 1998": 167, "Nissan Juke Hatchback 2012": 168, "Nissan Leaf Hatchback 2012": 169,
                "Nissan NV Passenger Van 2012": 170, "Plymouth Neon Coupe 1999": 171,
                "Porsche Panamera Sedan 2012": 172, "Ram C_V Cargo Van Minivan 2012": 173,
                "Rolls-Royce Ghost Sedan 2012": 174, "Rolls-Royce Phantom Drophead Coupe Convertible 2012": 175,
                "Rolls-Royce Phantom Sedan 2012": 176, "Scion xD Hatchback 2012": 177,
                "smart fortwo Convertible 2012": 178, "Spyker C8 Convertible 2009": 179, "Spyker C8 Coupe 2009": 180,
                "Suzuki Aerio Sedan 2007": 181, "Suzuki Kizashi Sedan 2012": 182, "Suzuki SX4 Hatchback 2012": 183,
                "Suzuki SX4 Sedan 2012": 184, "Tesla Model S Sedan 2012": 185, "Toyota 4Runner SUV 2012": 186,
                "Toyota Camry Sedan 2012": 187, "Toyota Corolla Sedan 2012": 188, "Toyota Sequoia SUV 2012": 189,
                "Volkswagen Beetle Hatchback 2012": 190, "Volkswagen Golf Hatchback 1991": 191,
                "Volkswagen Golf Hatchback 2012": 192, "Volvo 240 Sedan 1993": 193, "Volvo C30 Hatchback 2012": 194,
                "Volvo XC90 SUV 2007": 195}
    
    # gr.Image的图片格式为数组，需要转为图片后再进行处理
    print(type(image))
    image = Image.fromarray(image)
    category = classify_input(image)
    
    # 根据图片类别使用不同的模型进行后续工作
    if category == "flower":
        flower_model.load_state_dict(torch.load(flower_weight_path))
        flower_model.to("cuda:0").eval()
        # 定义提取平均池化层的模型
        feature_model = copy.deepcopy(flower_model)
        feature_model.fc = nn.Identity()  # 去除最后的全连接层，此时feature_model的输出为平均池化层
    else:
        car_model.load_state_dict(torch.load(car_weight_path))
        car_model.to("cuda:0").eval()
        feature_model = copy.deepcopy(car_model)
        feature_model.fc = nn.Identity()

    image_tensor = transform(image).to("cuda:0").unsqueeze(0)
    with torch.no_grad():
        output = feature_model(image_tensor)
        if category == "flower":
            kind = flower_model(image_tensor)
            print(kind)
            kind = kind.argmax(dim=1).item()
            for key, val in flower_dict.items():
                if val == kind:
                    Type = key
        else:
            kind = car_model(image_tensor).argmax(dim=1).item()
            for key, val in car_dict.items():
                if val == kind:
                    Type = key
        
        # 系统逻辑需要将哈希码显示在gradio界面上，gradio运行在cpu，而结果在gpu，需要取下来之后才能传递
        return output.cpu(), np.array2string(output.cpu().numpy()), category, category, Type


# 计算需要传入数据库图像的哈希码，以及其它数据库存储的信息
def calculate_feature_code_db(image, path):
    flower_dict = {"Bluebell": 0, "Buttercup": 1, "Colts'Foot": 2, "Cowslip": 3, "Crocus": 4, "Daffodil": 5, "Daisy": 6,
                   "Dandelion": 7, "Fritillary": 8, "Iris": 9, "LilyValley": 10, "Pansy": 11, "Snowdrop": 12,
                   "Sunflower": 13, "Tigerlily": 14, "Tulip": 15, "Windflower": 16}
    car_dict = {"Acura Integra Type R 2001": 0, "Acura RL Sedan 2012": 1, "Acura TL Sedan 2012": 2,
                "Acura TL Type-S 2008": 3, "Acura TSX Sedan 2012": 4, "Acura ZDX Hatchback 2012": 5,
                "AM General Hummer SUV 2000": 6, "Aston Martin V8 Vantage Convertible 2012": 7,
                "Aston Martin V8 Vantage Coupe 2012": 8, "Aston Martin Virage Convertible 2012": 9,
                "Aston Martin Virage Coupe 2012": 10, "Audi 100 Sedan 1994": 11, "Audi 100 Wagon 1994": 12,
                "Audi A5 Coupe 2012": 13, "Audi R8 Coupe 2012": 14, "Audi RS 4 Convertible 2008": 15,
                "Audi S4 Sedan 2007": 16, "Audi S4 Sedan 2012": 17, "Audi S5 Convertible 2012": 18,
                "Audi S5 Coupe 2012": 19, "Audi S6 Sedan 2011": 20, "Audi TT Hatchback 2011": 21,
                "Audi TT RS Coupe 2012": 22, "Audi TTS Coupe 2012": 23, "Audi V8 Sedan 1994": 24,
                "Bentley Arnage Sedan 2009": 25, "Bentley Continental Flying Spur Sedan 2007": 26,
                "Bentley Continental GT Coupe 2007": 27, "Bentley Continental GT Coupe 2012": 28,
                "Bentley Continental Supersports Conv. Convertible 2012": 29, "Bentley Mulsanne Sedan 2011": 30,
                "BMW 1 Series Convertible 2012": 31, "BMW 1 Series Coupe 2012": 32, "BMW 3 Series Sedan 2012": 33,
                "BMW 3 Series Wagon 2012": 34, "BMW 6 Series Convertible 2007": 35,
                "BMW ActiveHybrid 5 Sedan 2012": 36, "BMW M3 Coupe 2012": 37, "BMW M5 Sedan 2010": 38,
                "BMW M6 Convertible 2010": 39, "BMW X3 SUV 2012": 40, "BMW X5 SUV 2007": 41, "BMW X6 SUV 2012": 42,
                "BMW Z4 Convertible 2012": 43, "Bugatti Veyron 16.4 Convertible 2009": 44,
                "Bugatti Veyron 16.4 Coupe 2009": 45, "Buick Enclave SUV 2012": 46, "Buick Rainier SUV 2007": 47,
                "Buick Regal GS 2012": 48, "Buick Verano Sedan 2012": 49, "Cadillac CTS-V Sedan 2012": 50,
                "Cadillac Escalade EXT Crew Cab 2007": 51, "Cadillac SRX SUV 2012": 52,
                "Chevrolet Avalanche Crew Cab 2012": 53, "Chevrolet Camaro Convertible 2012": 54,
                "Chevrolet Cobalt SS 2010": 55, "Chevrolet Corvette Convertible 2012": 56,
                "Chevrolet Corvette Ron Fellows Edition Z06 2007": 57, "Chevrolet Corvette ZR1 2012": 58,
                "Chevrolet Express Cargo Van 2007": 59, "Chevrolet Express Van 2007": 60, "Chevrolet HHR SS 2010": 61,
                "Chevrolet Impala Sedan 2007": 62, "Chevrolet Malibu Hybrid Sedan 2010": 63,
                "Chevrolet Malibu Sedan 2007": 64, "Chevrolet Monte Carlo Coupe 2007": 65,
                "Chevrolet Silverado 1500 Classic Extended Cab 2007": 66,
                "Chevrolet Silverado 1500 Extended Cab 2012": 67, "Chevrolet Silverado 1500 Hybrid Crew Cab 2012": 68,
                "Chevrolet Silverado 1500 Regular Cab 2012": 69, "Chevrolet Silverado 2500HD Regular Cab 2012": 70,
                "Chevrolet Sonic Sedan 2012": 71, "Chevrolet Tahoe Hybrid SUV 2012": 72,
                "Chevrolet TrailBlazer SS 2009": 73, "Chevrolet Traverse SUV 2012": 74, "Chrysler 300 SRT-8 2010": 75,
                "Chrysler Aspen SUV 2009": 76, "Chrysler Crossfire Convertible 2008": 77,
                "Chrysler PT Cruiser Convertible 2008": 78, "Chrysler Sebring Convertible 2010": 79,
                "Chrysler Town and Country Minivan 2012": 80, "Daewoo Nubira Wagon 2002": 81,
                "Dodge Caliber Wagon 2007": 82, "Dodge Caliber Wagon 2012": 83, "Dodge Caravan Minivan 1997": 84,
                "Dodge Challenger SRT8 2011": 85, "Dodge Charger Sedan 2012": 86, "Dodge Charger SRT-8 2009": 87,
                "Dodge Dakota Club Cab 2007": 88, "Dodge Dakota Crew Cab 2010": 89, "Dodge Durango SUV 2007": 90,
                "Dodge Durango SUV 2012": 91, "Dodge Journey SUV 2012": 92, "Dodge Magnum Wagon 2008": 93,
                "Dodge Ram Pickup 3500 Crew Cab 2010": 94, "Dodge Ram Pickup 3500 Quad Cab 2009": 95,
                "Dodge Sprinter Cargo Van 2009": 96, "Eagle Talon Hatchback 1998": 97,
                "Ferrari 458 Italia Convertible 2012": 98, "Ferrari 458 Italia Coupe 2012": 99,
                "Ferrari California Convertible 2012": 100, "Ferrari FF Coupe 2012": 101, "FIAT 500 Abarth 2012": 102,
                "FIAT 500 Convertible 2012": 103, "Fisker Karma Sedan 2012": 104, "Ford E-Series Wagon Van 2012": 105,
                "Ford Edge SUV 2012": 106, "Ford Expedition EL SUV 2009": 107, "Ford F-150 Regular Cab 2007": 108,
                "Ford F-150 Regular Cab 2012": 109, "Ford F-450 Super Duty Crew Cab 2012": 110,
                "Ford Fiesta Sedan 2012": 111, "Ford Focus Sedan 2007": 112, "Ford Freestar Minivan 2007": 113,
                "Ford GT Coupe 2006": 114, "Ford Mustang Convertible 2007": 115, "Ford Ranger SuperCab 2011": 116,
                "Geo Metro Convertible 1993": 117, "GMC Acadia SUV 2012": 118, "GMC Canyon Extended Cab 2012": 119,
                "GMC Savana Van 2012": 120, "GMC Terrain SUV 2012": 121, "GMC Yukon Hybrid SUV 2012": 122,
                "Honda Accord Coupe 2012": 123, "Honda Accord Sedan 2012": 124, "Honda Odyssey Minivan 2007": 125,
                "Honda Odyssey Minivan 2012": 126, "HUMMER H2 SUT Crew Cab 2009": 127, "HUMMER H3T Crew Cab 2010": 128,
                "Hyundai Accent Sedan 2012": 129, "Hyundai Azera Sedan 2012": 130, "Hyundai Elantra Sedan 2007": 131,
                "Hyundai Elantra Touring Hatchback 2012": 132, "Hyundai Genesis Sedan 2012": 133,
                "Hyundai Santa Fe SUV 2012": 134, "Hyundai Sonata Hybrid Sedan 2012": 135,
                "Hyundai Sonata Sedan 2012": 136, "Hyundai Tucson SUV 2012": 137,
                "Hyundai Veloster Hatchback 2012": 138, "Hyundai Veracruz SUV 2012": 139,
                "Infiniti G Coupe IPL 2012": 140, "Infiniti QX56 SUV 2011": 141, "Isuzu Ascender SUV 2008": 142,
                "Jaguar XK XKR 2012": 143, "Jeep Compass SUV 2012": 144, "Jeep Grand Cherokee SUV 2012": 145,
                "Jeep Liberty SUV 2012": 146, "Jeep Patriot SUV 2012": 147, "Jeep Wrangler SUV 2012": 148,
                "Lamborghini Aventador Coupe 2012": 149, "Lamborghini Diablo Coupe 2001": 150,
                "Lamborghini Gallardo LP 570-4 Superleggera 2012": 151, "Lamborghini Reventon Coupe 2008": 152,
                "Land Rover LR2 SUV 2012": 153, "Land Rover Range Rover SUV 2012": 154,
                "Lincoln Town Car Sedan 2011": 155, "Maybach Landaulet Convertible 2012": 156,
                "Mazda Tribute SUV 2011": 157, "McLaren MP4-12C Coupe 2012": 158,
                "Mercedes-Benz 300-Class Convertible 1993": 159, "Mercedes-Benz C-Class Sedan 2012": 160,
                "Mercedes-Benz E-Class Sedan 2012": 161, "Mercedes-Benz S-Class Sedan 2012": 162,
                "Mercedes-Benz SL-Class Coupe 2009": 163, "Mercedes-Benz Sprinter Van 2012": 164,
                "MINI Cooper Roadster Convertible 2012": 165, "Mitsubishi Lancer Sedan 2012": 166,
                "Nissan 240SX Coupe 1998": 167, "Nissan Juke Hatchback 2012": 168, "Nissan Leaf Hatchback 2012": 169,
                "Nissan NV Passenger Van 2012": 170, "Plymouth Neon Coupe 1999": 171,
                "Porsche Panamera Sedan 2012": 172, "Ram C_V Cargo Van Minivan 2012": 173,
                "Rolls-Royce Ghost Sedan 2012": 174, "Rolls-Royce Phantom Drophead Coupe Convertible 2012": 175,
                "Rolls-Royce Phantom Sedan 2012": 176, "Scion xD Hatchback 2012": 177,
                "smart fortwo Convertible 2012": 178, "Spyker C8 Convertible 2009": 179, "Spyker C8 Coupe 2009": 180,
                "Suzuki Aerio Sedan 2007": 181, "Suzuki Kizashi Sedan 2012": 182, "Suzuki SX4 Hatchback 2012": 183,
                "Suzuki SX4 Sedan 2012": 184, "Tesla Model S Sedan 2012": 185, "Toyota 4Runner SUV 2012": 186,
                "Toyota Camry Sedan 2012": 187, "Toyota Corolla Sedan 2012": 188, "Toyota Sequoia SUV 2012": 189,
                "Volkswagen Beetle Hatchback 2012": 190, "Volkswagen Golf Hatchback 1991": 191,
                "Volkswagen Golf Hatchback 2012": 192, "Volvo 240 Sedan 1993": 193, "Volvo C30 Hatchback 2012": 194,
                "Volvo XC90 SUV 2007": 195}
    # gr.Image的图片格式为数组，需要转为图片后再进行处理
    print(type(image))
    image = Image.fromarray(image)
    category = classify_input(image)
    # 根据图片类别使用不同的模型进行后续工作
    if category == "flower":
        flower_model.load_state_dict(torch.load(flower_weight_path))
        flower_model.to("cuda:0").eval()
        # 定义提取平均池化层的模型
        feature_model = copy.deepcopy(flower_model)
        feature_model.fc = nn.Identity()  # 去除最后的全连接层，此时feature_model的输出为平均池化层
    else:
        car_model.load_state_dict(torch.load(car_weight_path))
        car_model.to("cuda:0").eval()
        feature_model = copy.deepcopy(car_model)
        feature_model.fc = nn.Identity()

    image_tensor = transform(image).to("cuda:0").unsqueeze(0)
    with torch.no_grad():
        output = feature_model(image_tensor)
        if category == "flower":
            kind = flower_model(image_tensor)
            print(kind)
            kind = kind.argmax(dim=1).item()
            for key, val in flower_dict.items():
                if val == kind:
                    Type = key
        else:
            kind = car_model(image_tensor).argmax(dim=1).item()
            for key, val in car_dict.items():
                if val == kind:
                    Type = key
        return output.cpu(), np.array2string(output.cpu().numpy()), category, Type, path


# gradio界面以block设计，运行后无法再创建组件，所以初始创建足够的组件，将其设为不可见
def retrival_show_message(search_num, feature_code, kind):
    # 初始化返回列表的值，根据数量设为可见和不可见
    visible_list = [gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                    gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                    gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                    gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                    gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                    gr.update(visible=True)]
    image_list = [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    category_list = [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    similarity_list = [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]

    # 可见和不可见由一个列表完成，gr.Image和两个gr.Text共同访问一个列表
    for i in range(search_num):
        visible_list[i] = gr.update(visible=True)

    coon = sqlite3.connect(db_path)
    cursor = coon.cursor()
    similar_list = []
    # 将输入系统图像的特征编码/哈希码与数据库存储的哈希码相比较，计算相似度
    if kind == "flower":
        cursor.execute("""SELECT id, feature_code FROM FLOWER""")
        result = cursor.fetchall()
        for row in result:
            id, feature_bytes = row
            feature_code_tensor = torch.load(io.BytesIO(feature_bytes))
            # 使用的余弦相似度
            similarity = F.cosine_similarity(feature_code, feature_code_tensor)
            similar_list.append([id, similarity.item()])
        sort_list = sorted(similar_list, key=lambda x: x[1], reverse=True)
        for i, j in zip(sort_list, range(search_num)):
            id, similar = i
            similarity_list[j] = similar
            cursor.execute("""SELECT image_path, category FROM FLOWER WHERE id = ?""", (id,))
            path, category = cursor.fetchone()
            # image_file = io.BytesIO(binary_date)
            # image = Image.open(image_file).convert("RGB")
            image_list[j] = path
            category_list[j] = category
    else:
        cursor.execute("""SELECT id, feature_code FROM CAR""")
        result = cursor.fetchall()
        for row in result:
            id, feature_bytes = row
            feature_code_tensor = torch.load(io.BytesIO(feature_bytes))
            similarity = F.cosine_similarity(feature_code, feature_code_tensor)
            similar_list.append([id, similarity.item()])
        sort_list = sorted(similar_list, key=lambda x: x[1], reverse=True)
        for i, j in zip(sort_list, range(search_num)):
            id, similar = i
            similarity_list[j] = similar
            cursor.execute("""SELECT image_path, category FROM CAR WHERE id = ?""", (id,))
            path, category = cursor.fetchone()
            # image_file = io.BytesIO(binary_date)
            # image = Image.open(image_file).convert("RGB")
            image_list[j] = path
            category_list[j] = category
    coon.close()

    return image_list[0], visible_list[0], category_list[0], visible_list[0], similarity_list[0], visible_list[0], \
           image_list[1], visible_list[1], category_list[1], visible_list[1], similarity_list[1], visible_list[1], \
           image_list[2], visible_list[2], category_list[2], visible_list[2], similarity_list[2], visible_list[2], \
           image_list[3], visible_list[3], category_list[3], visible_list[3], similarity_list[3], visible_list[3], \
           image_list[4], visible_list[4], category_list[4], visible_list[4], similarity_list[4], visible_list[4], \
           image_list[5], visible_list[5], category_list[5], visible_list[5], similarity_list[5], visible_list[5], \
           image_list[6], visible_list[6], category_list[6], visible_list[6], similarity_list[6], visible_list[6], \
           image_list[7], visible_list[7], category_list[7], visible_list[7], similarity_list[7], visible_list[7], \
           image_list[8], visible_list[8], category_list[8], visible_list[8], similarity_list[8], visible_list[8], \
           image_list[9], visible_list[9], category_list[9], visible_list[9], similarity_list[9], visible_list[9], \
           image_list[10], visible_list[10], category_list[10], visible_list[10], similarity_list[10], visible_list[10], \
           image_list[11], visible_list[11], category_list[11], visible_list[11], similarity_list[11], visible_list[11], \
           image_list[12], visible_list[12], category_list[12], visible_list[12], similarity_list[12], visible_list[12], \
           image_list[13], visible_list[13], category_list[13], visible_list[13], similarity_list[13], visible_list[13], \
           image_list[14], visible_list[14], category_list[14], visible_list[14], similarity_list[14], visible_list[14], \
           visible_list[15]


if __name__ == '__main__':
    img = Image.open("./test_image/000405.jpg").convert("RGB")
    classify_input(img)
