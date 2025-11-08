import json

def compare_json_captions():
    
    json_path1 = r"D:\大三资料\AIGC科研\IRRA\data\CUHK-PEDES\bear_noisy_data\reid_rw_with_test_noisy_2535_2641_63_917\reid_raw_0.8.json"
    # json_path2 = r"D:\大三资料\AIGC科研\IRRA\data\CUHK-PEDES\bear_noisy_data\reid_rw_with_test_noisy_2535_2641_63_917\reid_raw_0.2.json"
    json_path2 = r"D:\大三资料\AIGC科研\IRRA\data\CUHK-PEDES\bear_noisy_data\reid_rw_with_test_noisy_3007_1906_944_299\reid_raw_0.8.json"
    try:
        # 读取第一个JSON文件
        with open(json_path1, 'r', encoding='utf-8') as f1:
            data1 = json.load(f1)
        
        # 读取第二个JSON文件
        with open(json_path2, 'r', encoding='utf-8') as f2:
            data2 = json.load(f2)
        
        # 检查两个JSON文件的结构
        if not isinstance(data1, list) or not isinstance(data2, list):
            print("False")
            return False
        
        # 获取两个文件中split为"test"的样本
        test_samples1 = [item for item in data1 if item.get('split') == 'test']
        test_samples2 = [item for item in data2 if item.get('split') == 'test']
        
        # 检查test样本数量是否相同
        if len(test_samples1) != len(test_samples2):
            print("False")
            return False
        
        # 按text_id排序以确保对应关系正确
        test_samples1.sort(key=lambda x: x.get('text_id', 0))
        test_samples2.sort(key=lambda x: x.get('text_id', 0))
        
        # 逐个比较captions字段
        for sample1, sample2 in zip(test_samples1, test_samples2):
            captions1 = sample1.get('captions', [])
            captions2 = sample2.get('captions', [])
            
            # 检查captions列表长度和内容是否相同
            if len(captions1) != len(captions2):
                print("False")
                return False
            
            for cap1, cap2 in zip(captions1, captions2):
                if cap1 != cap2:
                    print("False")
                    return False
        
        print("True")
        return True
        
    except Exception as e:
        print(f"False")
        return False

# 使用示例
if __name__ == "__main__":
    compare_json_captions()