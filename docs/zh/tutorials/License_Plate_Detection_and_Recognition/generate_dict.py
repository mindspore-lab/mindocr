# 定义 provinces、alphabets 和 ads 列表
provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

# 合并所有字符并去重
unique_characters = set(provinces + alphabets + ads)

# 创建字典
unique_dict = {char: index for index, char in enumerate(sorted(unique_characters))}
print("length: " + str(len(unique_dict)))
# 打印并保存至 ccpd.txt
with open('ccpd.txt', 'w', encoding='utf-8') as file:
    for char, index in unique_dict.items():
        line = f"{char}:{index}\n"
        print(line.strip())  
        file.write(line)    
 